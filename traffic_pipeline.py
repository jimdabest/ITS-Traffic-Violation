import os
import csv
import time
import json
import yaml
import cv2
import torch
import numpy as np
from ultralytics import YOLO


def ensure_custom_tracker(tracker_path):
    """Tạo file cấu hình ByteTrack ổn định cho video dài."""
    if tracker_path == "custom_bytetrack.yaml":
        custom_cfg = (
            "tracker_type: bytetrack\n"
            "track_high_thresh: 0.25\n"
            "track_low_thresh: 0.10\n"
            "new_track_thresh: 0.30\n"
            "track_buffer: 120\n"
            "match_thresh: 0.80\n"
            "fuse_score: true\n"
        )
        with open(tracker_path, "w", encoding="utf-8") as f:
            f.write(custom_cfg)


def load_roi(roi_config_path, video_name):
    with open(roi_config_path, "r", encoding="utf-8") as f:
        roi_data = json.load(f)
    if isinstance(roi_data, dict):
        if video_name in roi_data:
            pts = roi_data[video_name]
            if isinstance(pts, dict) and "roi_polygon" in pts:
                pts = pts["roi_polygon"]
            return np.array(pts, dtype=np.int32)
        if "roi_polygon" in roi_data:
            return np.array(roi_data["roi_polygon"], dtype=np.int32)
    raise ValueError(f"Cấu trúc file {roi_config_path} không hợp lệ.")


def save_violations_csv(csv_path, records):
    v_fields = [
        "id", "video_name", "track_id", "vehicle_type",
        "timestamp_sec", "frame_id", "confidence", "crop_path", "full_path"
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=v_fields)
        writer.writeheader()
        writer.writerows(records)


def run_pipeline(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    video_paths = cfg["io"]["video_paths"]
    roi_config_path = cfg["io"]["roi_config_path"]
    crop_dir = cfg["io"]["crop_dir"]
    full_dir = cfg["io"]["full_dir"]
    tracks_csv_path = cfg["io"]["tracks_csv_path"]
    csv_prefix = cfg["io"]["csv_prefix"]
    save_video = cfg["io"].get("save_output_video", False)

    model_name = cfg["model"]["name"]
    imgsz = int(cfg["model"]["imgsz"])
    conf_thres = float(cfg["model"]["conf_thres"])
    iou_thres = float(cfg["model"]["iou_thres"])
    tracker_cfg = cfg["model"]["tracker"]
    use_augment = bool(cfg["model"].get("augment", False))
    use_half = bool(cfg["model"].get("half", False))
    use_agnostic = bool(cfg["model"].get("agnostic_nms", True))

    tracked_classes = cfg["logic"]["tracked_classes"]
    violating_cls = int(cfg["logic"]["violating_class_id"])
    violation_time_sec = float(cfg["logic"]["violation_time_sec"])
    lost_tolerance = int(cfg["logic"]["lost_track_tolerance"])
    allow_roi_gap_frames = int(cfg["logic"].get("allow_roi_gap_frames", 2))
    min_hits_ratio = float(cfg["logic"].get("min_hits_ratio", 0.4))

    ensure_custom_tracker(tracker_cfg)
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(full_dir, exist_ok=True)
    if os.path.dirname(tracks_csv_path):
        os.makedirs(os.path.dirname(tracks_csv_path), exist_ok=True)

    tracks_file = open(tracks_csv_path, mode="w", newline="", encoding="utf-8")
    track_fields = [
        "video_name", "frame_id", "timestamp_sec", "fps",
        "track_id", "class_id", "conf",
        "x1", "y1", "x2", "y2", "cx", "cy", "inside_roi"
    ]
    tracks_writer = csv.DictWriter(tracks_file, fieldnames=track_fields)
    tracks_writer.writeheader()

    violation_records = []
    violation_id = 1
    total_frames_all, total_time_all = 0, 0.0

    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"[CẢNH BÁO] Không tìm thấy file: {video_path}")
            continue

        video_name = os.path.basename(video_path)
        base_no_ext = os.path.splitext(video_name)[0]
        roi_polygon = load_roi(roi_config_path, video_name)

        model = YOLO(model_name)
        device = 0 if torch.cuda.is_available() else "cpu"
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        n_frames_thres = int(np.ceil(violation_time_sec * fps))

        writer = None
        if save_video:
            out_path = f"out_{video_name}"
            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            print(f"[INFO] Sẽ xuất video: {out_path}")

        track_states = {}
        logged_tracks = set()
        frame_id = 0
        t_start_video = time.perf_counter()

        print(f"\n--- Đang xử lý: {video_name} ---")
        print(f"    FPS: {fps:.2f} | Tổng frames: {total_video_frames} | Device: {device}")
        print(f"    Ngưỡng vi phạm: {n_frames_thres} frames ({violation_time_sec}s)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            current_sec = round(frame_id / fps, 2)
            
            # Nếu cần xuất video, tạo bản sao để vẽ. Nếu không, vẽ trực tiếp lên frame gốc
            # (giúp tiết kiệm RAM cho video dài)
            if save_video:
                annotated_frame = frame.copy()
                cv2.polylines(annotated_frame, [roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
            else:
                annotated_frame = frame

            results = model.track(
                frame,
                persist=True,
                tracker=tracker_cfg,
                classes=tracked_classes,
                conf=conf_thres,
                iou=iou_thres,
                imgsz=imgsz,
                augment=use_augment,
                quantize=16 if use_half else 32,
                agnostic_nms=use_agnostic,
                device=device,
                verbose=False
            )

            active_ids = set()

            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confs = results[0].boxes.conf.cpu().tolist()
                cls_ids = results[0].boxes.cls.int().cpu().tolist()

                for box, t_id, conf, cls_id in zip(boxes, track_ids, confs, cls_ids):
                    active_ids.add(t_id)
                    x1, y1, x2, y2 = map(int, box)
                    cx = int((x1 + x2) / 2)
                    cy = int(y2)

                    dist_to_roi = cv2.pointPolygonTest(roi_polygon, (float(cx), float(cy)), True)
                    inside_roi = 1 if dist_to_roi >= 0 else 0

                    tracks_writer.writerow({
                        "video_name": video_name, "frame_id": frame_id,
                        "timestamp_sec": f"{current_sec:.2f}", "fps": f"{fps:.2f}",
                        "track_id": t_id, "class_id": cls_id, "conf": f"{conf:.4f}",
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "cx": cx, "cy": cy, "inside_roi": inside_roi
                    })

                    # Vẽ box
                    if save_video:
                        color_box = (0, 0, 255) if inside_roi else (0, 255, 0)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_box, 2)
                        cv2.putText(annotated_frame, f"ID:{t_id} C:{cls_id}", (x1, max(15, y1-5)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 1)

                    if cls_id not in (violating_cls, 1):
                        continue

                    # Logic vi phạm
                    st = track_states.setdefault(t_id, {
                        "consecutive_frames": 0, "valid_detections": 0, "lost_frames": 0,
                        "start_timestamp": 0.0, "start_frame_id": 0, "conf_sum": 0.0, "roi_history": []
                    })

                    if inside_roi == 1:
                        if st["consecutive_frames"] == 0:
                            st["start_timestamp"] = current_sec
                            st["start_frame_id"] = frame_id
                            st["valid_detections"] = 1
                            st["consecutive_frames"] = 1
                            st["roi_history"] = [1]
                        else:
                            st["consecutive_frames"] += (1 + st["lost_frames"])
                            for _ in range(st["lost_frames"]):
                                st["roi_history"].append(0)
                            st["roi_history"].append(1)
                            st["valid_detections"] += 1
                        st["lost_frames"] = 0
                        st["conf_sum"] += conf
                    else:
                        if st["consecutive_frames"] > 0 and st["lost_frames"] < allow_roi_gap_frames:
                            st["lost_frames"] += 1
                            st["roi_history"].append(0)
                        else:
                            st["consecutive_frames"] = 0
                            st["valid_detections"] = 0
                            st["lost_frames"] = 0
                            st["conf_sum"] = 0.0
                            st["roi_history"] = []

                    window = st["roi_history"][-max(n_frames_thres, 1):]
                    hit_ratio = sum(window) / max(1, len(window))
                    min_hits_count = max(1, int(n_frames_thres * min_hits_ratio))
                    is_violating_now = (
                        st["consecutive_frames"] >= n_frames_thres 
                        and hit_ratio >= min_hits_ratio 
                        and st["valid_detections"] >= min_hits_count
                    )

                    if is_violating_now and t_id not in logged_tracks:
                        logged_tracks.add(t_id)
                        
                        # Vẽ nhãn VI PHAM lên video
                        if save_video:
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(annotated_frame, f"ID:{t_id} VI PHAM", (x1, max(15, y1-5)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        crop_rel = f"{crop_dir}/{base_no_ext}_t{t_id}_f{frame_id}.jpg"
                        full_rel = f"{full_dir}/{base_no_ext}_t{t_id}_f{frame_id}.jpg"
                        
                        crop_img = frame[max(0, y1-20):min(height, y2+20), max(0, x1-20):min(width, x2+20)]
                        cv2.imwrite(crop_rel, crop_img)
                        
                        if save_video:
                            cv2.imwrite(full_rel, annotated_frame)
                        else:
                            demo_frame = frame.copy()
                            cv2.polylines(demo_frame, [roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
                            cv2.rectangle(demo_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(demo_frame, f"ID:{t_id} VI PHAM", (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.imwrite(full_rel, demo_frame)

                        avg_conf = st["conf_sum"] / max(1, st["valid_detections"])
                        v_type = "motorcycle" if cls_id == 3 else ("bicycle" if cls_id == 1 else "motorcycle")
                        
                        violation_records.append({
                            "id": violation_id, "video_name": video_name, "track_id": t_id,
                            "vehicle_type": v_type, "timestamp_sec": f"{st['start_timestamp']:.2f}",
                            "frame_id": st["start_frame_id"], "confidence": f"{avg_conf:.2f}",
                            "crop_path": crop_rel, "full_path": full_rel,
                        })
                        violation_id += 1

            for t_id in list(track_states.keys()):
                if t_id not in active_ids:
                    st = track_states[t_id]
                    st["lost_frames"] += 1
                    if st["consecutive_frames"] > 0:
                        if st["lost_frames"] <= allow_roi_gap_frames:
                            st["roi_history"].append(0)
                        else:
                            st["consecutive_frames"] = 0
                            st["valid_detections"] = 0
                            st["conf_sum"] = 0.0
                            st["roi_history"] = []
                    
                    if st["lost_frames"] >= lost_tolerance:
                        del track_states[t_id]

            if frame_id % 300 == 0 or frame_id == total_video_frames:
                pct = (frame_id / total_video_frames * 100) if total_video_frames > 0 else 0
                print(f"  -> Frame {frame_id}/{total_video_frames} ({pct:.1f}%) | Vi phạm: {len(violation_records)}")

            # Ghi frame vào video (nếu bật save_video)
            if writer is not None:
                writer.write(annotated_frame)

        cap.release()
        if writer is not None:
            writer.release()

        elapsed = time.perf_counter() - t_start_video
        vid_fps = frame_id / elapsed if elapsed > 0 else 0.0
        total_frames_all += frame_id
        total_time_all += elapsed
        print(f"-> Hoàn tất {video_name} | {frame_id} frames | {vid_fps:.2f} FPS | Vi phạm: {len(violation_records)}")

    tracks_file.close()

    model_tag = os.path.splitext(os.path.basename(model_name))[0]
    main_csv = f"{csv_prefix}_{model_tag}_{violation_time_sec}.csv"
    save_violations_csv(main_csv, violation_records)

    overall_fps = total_frames_all / total_time_all if total_time_all > 0 else 0.0
    print("\n================ TỔNG KẾT ================")
    print(f"1. File tracks       : {tracks_csv_path}")
    print(f"2. File CSV vi phạm  : {main_csv} ({len(violation_records)} sự kiện)")
    print(f"3. Tốc độ trung bình : {overall_fps:.2f} FPS")
    print("==========================================")


if __name__ == "__main__":
    run_pipeline("config.yaml")