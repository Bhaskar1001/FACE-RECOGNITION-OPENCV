import cv2
import numpy as np
import os

size = 4
haar_cascade_path = r"D:\open cv\haarcascade_frontalface_default.xml"
datasets = 'datasets'  # Ensure this path is correct
print('Training...')

(images, labels, names, id) = ([], [], {}, 0)
(width, height) = (130, 100)  # Fixed size for all images

# Load training images
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        print(f"Checking folder: {subjectpath}")  # Debugging

        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            print(f"Processing file: {path}")  # Debugging

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                resized_img = cv2.resize(img, (width, height))  # Resize to fixed dimensions
                images.append(resized_img)
                labels.append(id)
            else:
                print(f"Warning: Failed to load image {path}")  # Debugging

        id += 1

 # Ensure at least one image exists before training
if len(images) == 0 or len(labels) == 0:
    print("Error: No valid training data found. Check the dataset folder!")
    exit()

# Convert to NumPy arrays
images = np.array(images, dtype=np.uint8)  # Ensure uniform dtype
labels = np.array(labels, dtype=np.int32)

# Train the model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

print("Training completed successfully!")


