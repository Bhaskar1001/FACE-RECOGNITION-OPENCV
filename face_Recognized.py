import cv2
import numpy as np
import os

size = 4
haar_file = r"D:\open cv\haarcascade_frontalface_default.xml"  # Use raw string
datasets = "datasets"  # Make sure this folder exists and has images
print("Training...")

(images, labels, names, id) = ([], [], {}, 0)
(width, height) = (130, 100)  # Fixed size for images

# Load images from dataset
for subdirs, dirs, files in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subject_path = os.path.join(datasets, subdir)
        for filename in os.listdir(subject_path):
            path = os.path.join(subject_path, filename)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Ensure grayscale
            if img is not None:
                img_resized = cv2.resize(img, (width, height))  # Resize
                images.append(img_resized)
                labels.append(id)
            else:
                print(f"Warning: Could not read {path}")

        id += 1

# Ensure at least one image is loaded
if len(images) == 0 or len(labels) == 0:
    print("Error: No training data found!")
    exit()

# Convert lists to NumPy arrays
images = np.array(images, dtype=np.uint8)  # Ensure uniform dtype
labels = np.array(labels, dtype=np.int32)

# Train the model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
print("Training completed successfully!")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)  # Use default camera
cnt = 0

# Start face recognition
while True:
    ret, im = webcam.read()
    if not ret:
        print("Error: Could not read from webcam.")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (width, height))

        prediction = model.predict(face_resized)
        confidence = prediction[1]  # Lower value = better match

        if confidence < 800:
            cv2.putText(im, f"{names[prediction[0]]} - {confidence:.0f}", (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
            print(f"Recognized: {names[prediction[0]]}")
            cnt = 0
        else:
            cnt += 1
            cv2.putText(im, "Unknown", (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if cnt > 100:
                print("Unknown Person Detected")
                cv2.imwrite("unknown.jpg", im)
                cnt = 0

    cv2.imshow("Face Recognition", im)
    key = cv2.waitKey(10)
    if key == 27:  # Press "Esc" to exit
        break

webcam.release()
cv2.destroyAllWindows()
