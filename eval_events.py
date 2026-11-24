#!/usr/bin/env python3
"""
Đánh giá sự kiện vi phạm: ghép cặp 1-1 theo video_name và timestamp_sec.
Chỉ dùng thư viện chuẩn Python.

CSV đầu vào cần có:
  video_name,timestamp_sec
Có thể có thêm vehicle_type. Dòng Ground Truth nên chỉ chứa sự kiện Vi_pham.

Ví dụ:
  python eval_events.py --gt ground_truth.csv --pred violations.csv --tolerance 2
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class Event:
    row_id: str
    video_name: str
    timestamp_sec: float
    vehicle_type: str = ""

def normalize_type(value: str) -> str:
    return (value or "").strip().lower().replace(" ", "_")

def load_events(path: str, id_prefix: str, only_violations: bool = False):
    events = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"{path}: CSV không có dòng tiêu đề.")

        fields = {name.strip().lower(): name for name in reader.fieldnames}
        required = ["video_name", "timestamp_sec"]
        missing = [x for x in required if x not in fields]

        if missing:
            raise ValueError(
                f"{path}: thiếu cột {missing}. ""Cần tối thiểu video_name,timestamp_sec.")

        for index, row in enumerate(reader, start=1):
            # Nếu là file Ground Truth có cột ket_luan,
            # bỏ dòng không phải Vi_pham.
            if only_violations and "ket_luan" in fields:
                verdict = (row.get(fields["ket_luan"]) or "").strip().lower()
                if verdict and verdict not in ("vi_pham", "vi phạm", "violation"):
                    continue
            raw_time = row.get(fields["timestamp_sec"], "")
            if raw_time is None or str(raw_time).strip() == "":
                continue
            try:
                timestamp = float(raw_time)
            except ValueError:
                raise ValueError(f"{path}, dòng {index + 1}: "
                    f"timestamp_sec không hợp lệ: {raw_time}")
            video_name = (row.get(fields["video_name"]) or "").strip()
            if not video_name:
                raise ValueError(
                    f"{path}, dòng {index + 1}: video_name bị trống.")
            vehicle_type = ""
            if "vehicle_type" in fields:
                vehicle_type = normalize_type(
                    row.get(fields["vehicle_type"], ""))
            row_id = ""
            for candidate in ("event_id", "id", "track_id"):
                if candidate in fields and row.get(fields[candidate]):
                    row_id = str(row[fields[candidate]]).strip()
                    break
            if not row_id:
                row_id = f"{id_prefix}{index:04d}"
            events.append(Event(row_id, video_name, timestamp, vehicle_type))
    return events

def types_compatible(a: str, b: str, require_same_type: bool) -> bool:
    if not require_same_type:
        return True
    # Loại không xác định được xem là wildcard để tránh loại nhầm sự kiện.
    unknown = {"", "khong_xac_dinh", "không_xác_định", "unknown"}
    if a in unknown or b in unknown:
        return True
    return a == b

def match_events(gt_events, pred_events, tolerance: float, require_same_type: bool):
    # Tạo toàn bộ cặp hợp lệ; chọn các cặp có độ lệch thời gian nhỏ nhất trước.
    candidates = []
    for gi, gt in enumerate(gt_events):
        for pi, pred in enumerate(pred_events):
            if gt.video_name != pred.video_name:
                continue
            if not types_compatible(gt.vehicle_type,pred.vehicle_type,require_same_type):
                continue
            delta = abs(gt.timestamp_sec - pred.timestamp_sec)
            if delta <= tolerance:
                candidates.append((delta, gi, pi))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))

    used_gt, used_pred, matches = set(), set(), []
    for delta, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gt_events[gi], pred_events[pi], delta))
        
    false_negatives = [e for i, e in enumerate(gt_events) if i not in used_gt]
    false_positives = [e for i, e in enumerate(pred_events) if i not in used_pred]
    return matches, false_positives, false_negatives

def safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0

def write_details(path: str, matches, fps, fns):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loai","video_name","gt_id","pred_id",
            "gt_timestamp_sec","pred_timestamp_sec","abs_delta_sec"])
        for gt, pred, delta in matches:
            writer.writerow(["TP",gt.video_name,gt.row_id,pred.row_id,
                f"{gt.timestamp_sec:.3f}",f"{pred.timestamp_sec:.3f}",f"{delta:.3f}"])
        for pred in fps:
            writer.writerow(["FP",pred.video_name,"",pred.row_id,"",
                f"{pred.timestamp_sec:.3f}",""])
        for gt in fns:
            writer.writerow(["FN",gt.video_name,gt.row_id,"",
                f"{gt.timestamp_sec:.3f}","",""])

def main():
    parser = argparse.ArgumentParser(description="Tính TP/FP/FN, Precision, Recall, F1.")
    parser.add_argument("--gt", required=True, help="CSV Ground Truth")
    parser.add_argument("--pred", required=True, help="CSV dự đoán từ pipeline")
    parser.add_argument("--tolerance",type=float,default=2.0,
        help="Sai lệch thời gian tối đa, giây (mặc định 2.0)")
    parser.add_argument("--same-type",action="store_true",
        help="Yêu cầu loại xe tương thích khi ghép cặp")

    parser.add_argument("--details",default="eval_details.csv",
        help="File CSV chi tiết TP/FP/FN")
    args = parser.parse_args()
    
    gt = load_events(args.gt, "GT", only_violations=True)
    pred = load_events(args.pred, "PRED", only_violations=False)
    matches, fp_events, fn_events = match_events(gt, pred, args.tolerance, args.same_type)
    
    tp, fp, fn = len(matches), len(fp_events), len(fn_events)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    print(f"So su kien Ground Truth: {len(gt)}")
    print(f"So su kien du doan:      {len(pred)}")
    print(f"TP={tp}, FP={fp}, FN={fn}")
    print(f"Precision={precision:.4f}")
    print(f"Recall={recall:.4f}")
    print(f"F1-score={f1:.4f}")
    print(f"Chi tiet: {args.details}")

    write_details(args.details, matches, fp_events, fn_events)

if __name__ == "__main__":
    main()
