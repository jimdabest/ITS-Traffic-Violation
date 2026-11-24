# Hệ Thống Phát Hiện Xe Máy Lấn Làn Ô Tô (YOLOv8 + ByteTrack)

Dự án môn Hệ thống Giao thông Thông minh (ITS) — Phát hiện và ghi nhận sự kiện xe máy đi vào làn ô tô dựa trên vùng quan tâm (ROI) và thuật toán theo vết đa đối tượng.

## 1. Cấu trúc thư mục

```text
ITS_Pipeline/
├── traffic_pipeline.py     # Pipeline chính: nhận diện, theo vết, xét ROI, xuất CSV
├── auto_run_gd3.py         # Script chạy tự động 2 mô hình (v8n, v8s) × 3 ngưỡng thời gian
├── config.yaml             # Cấu hình mô hình, ngưỡng ROI, đường dẫn đầu vào/ra
├── roi_config.json         # Tọa độ đa giác vùng cấm (ROI) cho từng video
├── custom_bytetrack.yaml   # Cấu hình tinh chỉnh ByteTrack (tự sinh khi chạy)
├── requirements.txt        # Danh sách thư viện cần thiết
└── README.md               # Hướng dẫn cài đặt và vận hành
```

## 2. Yêu cầu môi trường

- Python 3.9 – 3.11
- GPU hỗ trợ CUDA (khuyến nghị) hoặc CPU.

## 3. Hướng dẫn cài đặt

```bash
# 1. Tạo và kích hoạt môi trường ảo (khuyến nghị)
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

## 4. Hướng dẫn chạy

1. Đặt file video đầu vào (ví dụ: `phamvandong_video_01.mp4`) vào thư mục gốc và cập nhật tên file trong `config.yaml` nếu cần.
2. Đảm bảo file `roi_config.json` có cấu hình đa giác tương ứng với tên video.
3. Chạy pipeline đơn lẻ:

```bash
python traffic_pipeline.py
```

4. Hoặc chạy tự động toàn bộ thực nghiệm (YOLOv8n & YOLOv8s với 3 ngưỡng 0.3s, 0.5s, 0.8s):

```bash
python auto_run_gd3.py
```

## 5. Đầu ra (Output)

- `tracks_<model>.csv`: Dữ liệu theo vết thô theo từng khung hình.
- `violations_<model>_<time>.csv`: Danh sách sự kiện vi phạm (chuẩn 9 cột: `id`, `video_name`, `track_id`, `vehicle_type`, `timestamp_sec`, `frame_id`, `confidence`, `crop_path`, `full_path`).
- Thư mục `crop/` và `full/`: Ảnh minh chứng cắt cận cảnh và toàn cảnh tại thời điểm ghi nhận vi phạm.