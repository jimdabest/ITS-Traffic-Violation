import json

# Đọc nội dung file traffic_pipeline_v1.py hiện tại
with open("traffic_pipeline_v1.py", "r", encoding="utf-8") as f:
    pipeline_code = f.read()

notebook = {
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {"name": "colab_backup.ipynb"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## BƯỚC 1: Cài đặt thư viện"]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": ["!pip install -q ultralytics opencv-python lapx"]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## BƯỚC 2: Upload Video và file cấu hình ROI\n", "*(Chọn 2 file: `demo_clip.mp4` và `roi_config.json` từ máy tính)*"]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "from google.colab import files\n",
        "print('Vui lòng chọn demo_clip.mp4 và roi_config.json:')\n",
        "uploaded = files.upload()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## BƯỚC 3: Chạy Pipeline v1.1"]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [pipeline_code]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## BƯỚC 4: Đóng gói và Tải kết quả về máy"]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import os\n",
        "from google.colab import files\n\n",
        "# Nén toàn bộ kết quả vào 1 file zip\n",
        "!zip -r ketqua_GD1.zip output_demo_v1.mp4 violations_yolov8s_0.5.csv crop/ full/\n\n",
        "# Tự động tải về máy\n",
        "if os.path.exists('ketqua_GD1.zip'):\n",
        "    files.download('ketqua_GD1.zip')\n",
        "else:\n",
        "    print('Lỗi: Không tìm thấy file kết quả để tải về.')"
      ]
    }
  ]
}

with open("colab_backup.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print("Đã tạo thành công file colab_backup.ipynb!")