import cv2
import os

# Ensure the dataset path is correct
path = r"D:\dataset\bhaskar"
os.makedirs(path, exist_ok=True)  # Ensure directory exists
print(f"Directory '{path}' created successfully.")

# Load Haar Cascade model
haar_cascade_path = r"D:\open cv\haarcascade_frontalface_default.xml"
if not os.path.exists(haar_cascade_path):
    print("Error: Haar cascade file not found!")
    exit()

face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Define image size
(width, height) = (130, 100)

# Open webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not access webcam!")
    exit()

count = 1
while count <= 30:
    ret, im = webcam.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        # Ensure correct saving path
        filename = os.path.join(path, f"{count}.png")
        print(f"Saving image: {filename}")  # Debugging print
        cv2.imwrite(filename, face_resize)

        count += 1

    cv2.imshow('OpenCV', im)
    
    # Exit when 'Esc' is pressed
    key = cv2.waitKey(10)
    if key == 27 or count > 30:
        break

# Cleanup
webcam.release()
cv2.destroyAllWindows()
