import os

import cv2
folder_path = 'pic2'  # Thay đổi đường dẫn này
output_folder = 'pic3'  # Thư mục để lưu ảnh đã nhận dạng

# Đường dẫn tới các tệp mô hình
face_pbtxt = "models/opencv_face_detector.pbtxt"
face_pb = "models/opencv_face_detector_uint8.pb"
age_prototxt = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
gender_prototxt = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"

MODEL_MEAN_VALUES = [104, 117, 123]

# Tải mô hình
try:
    face_net = cv2.dnn.readNet(face_pb, face_pbtxt)
    age_net = cv2.dnn.readNet(age_model, age_prototxt)
    gender_net = cv2.dnn.readNet(gender_model, gender_prototxt)
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Danh sách phân loại tuổi và giới tính
age_classifications = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_classifications = ['Male', 'Female']

# =========================== CHỨC NĂNG 1: TÍNH ĐỘ CHÍNH XÁC VỚI CSV ===========================

def evaluate_accuracy_from_folders(input_folder, output_folder, display_size=(400, 600)):
    true_predictions = 0
    false_predictions = 0
    total_faces_detected = 0
    correct_images = []  # Lưu ảnh đã đoán đúng

    # Lặp qua các ảnh trong thư mục input
    for filename in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, f"correct_{filename}")

        # Đọc ảnh từ thư mục input
        input_img = cv2.imread(input_file_path)
        output_img = cv2.imread(output_file_path)

        if input_img is None or output_img is None:
            print(f"Lỗi khi đọc ảnh {filename}")
            continue

        img_h, img_w = input_img.shape[:2]
        blob = cv2.dnn.blobFromImage(input_img, 0.3, (300, 300), MODEL_MEAN_VALUES, True, False)

        face_net.setInput(blob)
        detected_faces = face_net.forward()

        for i in range(detected_faces.shape[2]):
            confidence = detected_faces[0, 0, i, 2]
            if confidence > 0.99:  # Ngưỡng tin cậy
                x1 = int(detected_faces[0, 0, i, 3] * img_w)
                y1 = int(detected_faces[0, 0, i, 4] * img_h)
                x2 = int(detected_faces[0, 0, i, 5] * img_w)
                y2 = int(detected_faces[0, 0, i, 6] * img_h)
                total_faces_detected += 1

                # Cắt vùng khuôn mặt từ ảnh
                face_img_input = input_img[y1:y2, x1:x2]
                face_img_output = output_img[y1:y2, x1:x2]

                # Resize ảnh (ví dụ resize thành 300x300)
                face_img_input_resized = cv2.resize(face_img_input, (300, 300))
                face_img_output_resized = cv2.resize(face_img_output, (300, 300))

                if face_img_input.size != 0 and face_img_output.size != 0:
                    # Dự đoán giới tính cho ảnh từ thư mục input
                    input_blob = cv2.dnn.blobFromImage(face_img_input_resized, 1.0, (227, 227), MODEL_MEAN_VALUES, True)
                    gender_net.setInput(input_blob)
                    gender_prediction = gender_net.forward()
                    predicted_gender_input = gender_classifications[gender_prediction[0].argmax()]

                    # Dự đoán độ tuổi cho ảnh từ thư mục input
                    age_net.setInput(input_blob)
                    age_prediction = age_net.forward()
                    predicted_age_class_input = age_classifications[age_prediction[0].argmax()]

                    # Dự đoán giới tính cho ảnh từ thư mục output
                    output_blob = cv2.dnn.blobFromImage(face_img_output_resized, 1.0, (227, 227), MODEL_MEAN_VALUES,
                                                        True)
                    gender_net.setInput(output_blob)
                    gender_prediction_output = gender_net.forward()
                    predicted_gender_output = gender_classifications[gender_prediction_output[0].argmax()]

                    # Dự đoán độ tuổi cho ảnh từ thư mục output
                    age_net.setInput(output_blob)
                    age_prediction_output = age_net.forward()
                    predicted_age_class_output = age_classifications[age_prediction_output[0].argmax()]

                    # So sánh kết quả giữa input và output
                    if (predicted_gender_input == predicted_gender_output) and (
                            predicted_age_class_input == predicted_age_class_output):
                        true_predictions += 1
                        # Khoanh vùng khuôn mặt đã nhận diện đúng trong ảnh input và output
                        cv2.rectangle(input_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(input_img, f'{predicted_gender_input}, {predicted_age_class_input}',
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        # Resize ảnh về kích thước mong muốn trước khi hiển thị
                        resized_img = cv2.resize(input_img, display_size)
                        # Thêm ảnh đã nhận diện đúng vào danh sách
                        correct_images.append(resized_img)
                    else:
                        false_predictions += 1

    # Tính độ chính xác
    if total_faces_detected > 0:
        accuracy = (true_predictions / total_faces_detected) * 100
        error_rate = 100 - accuracy
        print(f"Tổng số khuôn mặt phát hiện: {total_faces_detected}")
        print(f"Dự đoán đúng: {true_predictions}")
        print(f"Dự đoán sai: {false_predictions}")
        print(f"Độ chính xác (Accuracy): {accuracy:.2f}%")
        print(f"Tỷ lệ lỗi (Error Rate): {error_rate:.2f}%")
    else:
        print("Không phát hiện được khuôn mặt nào.")

    # Hiển thị ảnh đã đoán đúng
    if correct_images:
        for i, img in enumerate(correct_images):
            cv2.imshow(f"Correct Prediction {i + 1}", img)
        cv2.waitKey(0)  # Chờ người dùng nhấn phím
        cv2.destroyAllWindows()  # Đóng tất cả cửa sổ ảnh
    else:
        print("Không có ảnh nào đoán đúng.")



# Đường dẫn tới thư mục chứa ảnh





# =========================== CHỨC NĂNG 2: NHẬN DẠNG QUA CAMERA ===========================

def recognize_age_and_gender_from_camera():
    """
    Hàm nhận dạng tuổi và giới tính qua webcam
    """
    # Mở webcam
    cap = cv2.VideoCapture(0)  # ID 0 cho webcam mặc định

    if not cap.isOpened():
        print("Error: Webcam không thể mở được.")
        exit()

    print("Nhấn 'q' để thoát.")

    # Vòng lặp xử lý video
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc khung hình từ webcam.")
            break

        img_h, img_w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), MODEL_MEAN_VALUES, True, False)

        # Phát hiện khuôn mặt
        face_net.setInput(blob)
        detected_faces = face_net.forward()

        face_bounds = []
        for i in range(detected_faces.shape[2]):
            confidence = detected_faces[0, 0, i, 2]
            if confidence > 0.99:  # Ngưỡng độ tin cậy
                x1 = int(detected_faces[0, 0, i, 3] * img_w)
                y1 = int(detected_faces[0, 0, i, 4] * img_h)
                x2 = int(detected_faces[0, 0, i, 5] * img_w)
                y2 = int(detected_faces[0, 0, i, 6] * img_h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                face_bounds.append([x1, y1, x2, y2])

        # Xử lý từng khuôn mặt
        for face_bound in face_bounds:
            try:
                # Cắt vùng khuôn mặt
                face_img = frame[max(0, face_bound[1] - 15):min(face_bound[3] + 15, img_h),
                                 max(0, face_bound[0] - 15):min(face_bound[2] + 15, img_w)]

                # Chuyển thành blob
                blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, True)

                # Dự đoán giới tính
                gender_net.setInput(blob)
                gender_prediction = gender_net.forward()
                gender = gender_classifications[gender_prediction[0].argmax()]

                # Dự đoán tuổi
                age_net.setInput(blob)
                age_prediction = age_net.forward()
                age = age_classifications[age_prediction[0].argmax()]

                # Ghi thông tin lên khung hình
                label = f'{gender}, {age}'
                cv2.putText(frame, label, (face_bound[0], face_bound[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            except Exception as e:
                print(f"Lỗi khi xử lý khuôn mặt: {e}")
                continue

        # Hiển thị khung hình
        cv2.imshow('Webcam - Age & Gender Detection', frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()


# ========================================== MAIN ==========================================

print("Chọn chức năng:")
print("1. Tính độ chính xác từ folder ảnh nhận dạng 2")
print("2. Nhận dạng qua camera")

choice = input("Nhập số chức năng bạn muốn chọn (1 hoặc 2): ")

if choice == '1':
    # Để tính độ chính xác từ file CSV

    evaluate_accuracy_from_folders(folder_path, output_folder)
elif choice == '2':
    recognize_age_and_gender_from_camera()
