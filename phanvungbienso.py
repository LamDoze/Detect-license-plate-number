import cv2
import numpy as np

# Đọc ảnh
img = cv2.imread('Images/2.webp')

# Chuyển sang ảnh xám và làm mờ
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)

# Phát hiện cạnh
edge = cv2.Canny(blurred, 50, 200)

# Tìm đường viền
contours, _ = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

# Biến lưu vùng biển số đã cắt
license_plate_img = None

for c in contours:
    perimeter = cv2.arcLength(c, True)
    approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    
    # Nếu có 4 cạnh
    if len(approximation) == 4:
        x, y, w, h = cv2.boundingRect(approximation)
        
        # Cắt vùng biển số
        license_plate_img = img[y:y+h, x:x+w]
        
        # Vẽ viền biển số lên ảnh gốc
        cv2.drawContours(img, [approximation], -1, (0, 255, 0), 3)
        break

# Nếu vùng biển số được cắt thành công, phóng to và chèn lên ảnh gốc
if license_plate_img is not None:
    # Phóng to vùng biển số
    scale = 3  # Tăng kích thước lên 3 lần
    height, width = license_plate_img.shape[:2]
    enlarged_license_plate = cv2.resize(license_plate_img, (width * scale, height * scale), interpolation=cv2.INTER_LINEAR)
    
    # Vị trí để chèn ảnh phóng to (góc trên bên trái)
    x_offset, y_offset = 10, 10  # Khoảng cách từ góc trên trái
    x_end = x_offset + enlarged_license_plate.shape[1]
    y_end = y_offset + enlarged_license_plate.shape[0]
    
    # Kiểm tra xem kích thước có vượt quá ảnh gốc không
    if x_end <= img.shape[1] and y_end <= img.shape[0]:
        img[y_offset:y_end, x_offset:x_end] = enlarged_license_plate
    else:
        print("Biển số phóng to vượt kích thước ảnh gốc, không thể chèn lên.")
else:
    print("Không phát hiện được biển số.")

# Hiển thị ảnh kết quả
cv2.imshow('Hien thi', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
