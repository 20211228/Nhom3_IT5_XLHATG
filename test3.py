import cv2

# Load Image
image = cv2.imread('download.jpg')
image = cv2.resize(image, (720, 640))

# Model Paths
face_pbtxt = "models/opencv_face_detector.pbtxt"
face_pb = "models/opencv_face_detector_uint8.pb"
age_prototxt = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
gender_prototxt = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"
MODEL_MEAN_VALUES = [104, 117, 123]

# Load Models
face_net = cv2.dnn.readNet(face_pb, face_pbtxt)
age_net = cv2.dnn.readNet(age_model, age_prototxt)
gender_net = cv2.dnn.readNet(gender_model, gender_prototxt)

# Setup Classifications
age_classes = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_classes = ['Male', 'Female']

# Create a copy of the image
output_image = image.copy()

# Preprocess the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), MODEL_MEAN_VALUES, True, False)

# Detect faces
face_net.setInput(blob)
detections = face_net.forward()

# Loop through detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.7:  # Set confidence threshold
        x1 = int(detections[0, 0, i, 3] * image.shape[1])
        y1 = int(detections[0, 0, i, 4] * image.shape[0])
        x2 = int(detections[0, 0, i, 5] * image.shape[1])
        y2 = int(detections[0, 0, i, 6] * image.shape[0])

        # Extract the face
        face = image[max(0, y1):min(y2, image.shape[0]), max(0, x1):min(x2, image.shape[1])]

        # Skip if face region is too small
        if face.shape[0] < 10 or face.shape[1] < 10:
            continue

        # Preprocess the face for gender and age detection
        face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=True)

        # Predict Gender
        gender_net.setInput(face_blob)
        gender_preds = gender_net.forward()
        gender = gender_classes[gender_preds[0].argmax()]

        # Predict Age
        age_net.setInput(face_blob)
        age_preds = age_net.forward()
        age = age_classes[age_preds[0].argmax()]

        # Draw bounding box and label
        label = f"{gender}, {age}"
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# Display the output
cv2.imshow('Detected Faces', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
