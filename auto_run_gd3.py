import os
import csv
import yaml
import numpy as np

# Chỉ import duy nhất hàm chạy chính từ file traffic_pipeline
from traffic_pipeline import run_pipeline

def compute_offline_violations(tracks_csv_path, violation_time_sec=0.1, lost_tolerance=30, violating_cls=3, allow_roi_gap_frames=2, min_hits_ratio=0.4):
    by_video = {}
    if not os.path.exists(tracks_csv_path):
        print(f"[LỖI] Không tìm thấy file track thô: {tracks_csv_path}")
        return []
        
    with open(tracks_csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vname = row["video_name"]
            frame_id = int(row["frame_id"])
            fps = float(row["fps"])
            by_video.setdefault(vname, {"fps": fps, "frames": {}})
            by_video[vname]["frames"].setdefault(frame_id, []).append(row)

    offline_records = []
    vid_event_id = 1

    for vname, data in by_video.items():
        fps = data["fps"]
        n_frames_thres = int(np.ceil(violation_time_sec * fps))
        frames_dict = data["frames"]
        if not frames_dict:
            continue

        max_frame = max(frames_dict.keys())
        track_states = {}
        logged_tracks = set()

        for f_id in range(1, max_frame + 1):
            rows = frames_dict.get(f_id, [])
            active_ids = set()

            for r in rows:
                cls_id = int(r["class_id"])
                t_id = int(r["track_id"])
                active_ids.add(t_id)

                if cls_id not in (violating_cls, 1, 0):
                    continue

                inside_roi = int(r["inside_roi"]) == 1
                conf = float(r["conf"])
                current_sec = round(f_id / fps, 2)

                if t_id not in track_states:
                    track_states[t_id] = {
                        "consecutive_frames": 0, "valid_detections": 0, "lost_frames": 0,
                        "start_timestamp": 0.0, "start_frame_id": 0, "conf_sum": 0.0,
                        "roi_history": []
                    }

                st = track_states[t_id]

                if inside_roi:
                    if st["consecutive_frames"] == 0:
                        st["start_timestamp"] = current_sec
                        st["start_frame_id"] = f_id
                        st["valid_detections"] = 0
                        st["consecutive_frames"] = 1
                        st["roi_history"] = [1]
                    else:
                        st["consecutive_frames"] += (1 + st["lost_frames"])
                        st["roi_history"].append(1)
                    st["valid_detections"] += 1
                    st["lost_frames"] = 0
                    st["conf_sum"] += conf
                else:
                    if st["consecutive_frames"] > 0 and st["lost_frames"] <= allow_roi_gap_frames:
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
                is_violating = (st["consecutive_frames"] >= n_frames_thres and hit_ratio >= min_hits_ratio)

                if is_violating and t_id not in logged_tracks:
                    logged_tracks.add(t_id)
                    avg_conf = st["conf_sum"] / max(1, st["valid_detections"])
                    base_no_ext = os.path.splitext(vname)[0]
                    offline_records.append({
                        "id": vid_event_id, "video_name": vname, "track_id": t_id, "vehicle_type": "motorcycle",
                        "timestamp_sec": f"{st['start_timestamp']:.2f}", "frame_id": st["start_frame_id"],
                        "confidence": f"{avg_conf:.2f}", "crop_path": f"crop/{base_no_ext}_t{t_id}_f{f_id}.jpg",
                        "full_path": f"full/{base_no_ext}_t{t_id}_f{f_id}.jpg",
                    })
                    vid_event_id += 1

            for t_id, st in list(track_states.items()):
                if t_id not in active_ids and st["consecutive_frames"] > 0:
                    st["lost_frames"] += 1
                    if st["lost_frames"] >= lost_tolerance:
                        st["consecutive_frames"] = 0
                        st["valid_detections"] = 0
                        st["conf_sum"] = 0.0
                        st["roi_history"] = []

    return offline_records

def save_violations_csv(csv_path, records):
    v_fields = ["id", "video_name", "track_id", "vehicle_type", "timestamp_sec", "frame_id", "confidence", "crop_path", "full_path"]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=v_fields)
        writer.writeheader()
        writer.writerows(records)

def generate_all_reports():
    models = ["yolov8n.pt", "yolov8s.pt"]
    eval_times = [0.3, 0.5, 0.8]
    
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    for model in models:
        print(f"\n==========================================")
        print(f"🚀 ĐANG CHẠY SUY LUẬN MÔ HÌNH: {model}")
        print(f"==========================================")
        
        cfg["model"]["name"] = model
        model_tag = model.split('.')[0]
        cfg["io"]["tracks_csv_path"] = f"tracks_{model_tag}.csv"
        cfg["io"]["save_output_video"] = False # Đảm bảo chạy tốc độ cao
        
        with open("config_temp.yaml", "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
            
        # 1. Chạy 1 LẦN DUY NHẤT để xuất ra file track
        run_pipeline("config_temp.yaml")
        
        # 2. Xử lý Offline để trích xuất ra 3 file kết quả
        track_file = cfg["io"]["tracks_csv_path"]
        for t in eval_times:
            print(f"  -> Đang trích xuất dữ liệu vi phạm với ngưỡng {t} giây...")
            records = compute_offline_violations(
                tracks_csv_path=track_file, 
                violation_time_sec=t, 
                lost_tolerance=30,
                violating_cls=3, 
                allow_roi_gap_frames=2, 
                min_hits_ratio=0.4
            )
            out_name = f"violations_{model_tag}_{t}.csv"
            save_violations_csv(out_name, records)
            print(f"  ✅ Đã lưu file: {out_name} ({len(records)} sự kiện)")
            
    # Dọn dẹp file rác
    if os.path.exists("config_temp.yaml"):
        os.remove("config_temp.yaml")
        
    print("\n🎉 HOÀN THÀNH TOÀN BỘ YÊU CẦU CỦA STT 3! HÃY GỬI FILE CHO TV4.")

if __name__ == "__main__":
    generate_all_reports()