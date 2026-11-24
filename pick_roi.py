import cv2
import json

VIDEO_PATH = "phamvandong_video_01.mp4"
ROI_CONFIG_PATH = "roi_config.json"
points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        print(f"Đã chọn điểm {len(points)}: [{x}, {y}]")

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError(f"Không đọc được {VIDEO_PATH}. Hãy kiểm tra lại tên file!")

h_goc, w_goc = frame.shape[:2]

# Cố định chiều CAO hiển thị là 800 pixel, tính chiều RỘNG theo tỷ lệ để vừa màn hình
h_moi = 800
w_moi = int(w_goc * (h_moi / h_goc))

cv2.namedWindow("Chon 4 diem ROI (Bam Q de thoat)", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("Chon 4 diem ROI (Bam Q de thoat)", w_moi, h_moi)
cv2.setMouseCallback("Chon 4 diem ROI (Bam Q de thoat)", mouse_callback)

print("Vui lòng click 4 điểm tạo thành đa giác ROI (ưu tiên chọn theo chiều kim đồng hồ).")

while True:
    temp = frame.copy()
    for idx, pt in enumerate(points):
        cv2.circle(temp, tuple(pt), 6, (0, 0, 255), -1)
        cv2.putText(temp, str(idx + 1), (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(temp, tuple(points[i]), tuple(points[i + 1]), (0, 255, 0), 2)
            
    if len(points) == 4:
        cv2.line(temp, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)

    cv2.imshow("Chon 4 diem ROI (Bam Q de thoat)", temp)
    
    # Thoát nếu nhấn phím 'q' hoặc đã chọn đủ 4 điểm
    if cv2.waitKey(20) & 0xFF == ord('q') or len(points) == 4:
        break

cv2.waitKey(500)
cv2.destroyAllWindows()

# Lưu tọa độ vào file roi_config.json
if len(points) == 4:
    roi_data = {"roi_polygon": points}
    with open(ROI_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(roi_data, f, indent=2)
    print(f"\nThành công! Đã lưu tọa độ ROI vào {ROI_CONFIG_PATH}:")
    print(json.dumps(roi_data, indent=2))
else:
    print("\nChưa chọn đủ 4 điểm. Hủy lưu file ROI.")