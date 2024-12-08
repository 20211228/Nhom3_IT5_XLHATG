import cv2

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
        if confidence > 0.7:  # Ngưỡng độ tin cậy
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
