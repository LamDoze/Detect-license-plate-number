🛠 Chức năng nổi bật

  Phát hiện biển số xe: Tìm và vẽ viền biển số xe trên ảnh.
  Cắt biển số xe: Xác định vùng chứa biển số và cắt ra.
  Phóng to biển số: Kích thước biển số được phóng to để dễ dàng quan sát.
  Chèn biển số phóng to lên ảnh gốc: Đặt ảnh phóng to ở góc trên bên trái (nếu còn chỗ).

🏗 Cách hoạt động

  Đọc ảnh gốc: Tải hình ảnh của bạn từ thư mục Images.
  Chuyển sang ảnh xám và làm mờ: Loại bỏ nhiễu, chuẩn bị cho việc phát hiện cạnh.
  Phát hiện cạnh bằng Canny: Tìm ra các đường viền quan trọng.
  Tìm đường viền và phát hiện hình chữ nhật: Dựa trên hình dạng của biển số xe (4 cạnh).
  Xử lý vùng biển số:
      Cắt biển số từ ảnh gốc.
      Phóng to và kiểm tra kích thước trước khi chèn lên ảnh.
📷 Demo

![image](https://github.com/user-attachments/assets/26b14e95-cb59-4f93-8cb2-2c6a5568627f)

🔥 Mẹo sử dụng

  Định dạng ảnh: Hỗ trợ các định dạng phổ biến như .jpg, .png, .webp.
  Điều chỉnh độ nhạy: Nếu không phát hiện được biển số, hãy thử thay đổi các tham số trong hàm cv2.Canny.
  Thử nghiệm nhiều ảnh: Một số ảnh có độ nhiễu cao hoặc ánh sáng kém có thể không được xử lý tốt.

💡 Ý tưởng mở rộng

  Tự động nhận dạng ký tự biển số bằng OCR.
  Phân loại biển số theo vùng/khu vực.
  Tích hợp vào hệ thống camera giám sát.
