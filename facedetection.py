import urllib.request
import os
import cv2
import random
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Download and extract the Haar Cascade dataset for face detection
haar_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
urllib.request.urlretrieve(haar_url, 'haarcascade_frontalface_default.xml')

# Load the dataset of faces
face_dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a function to detect faces in an image
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_dataset.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

# Load the Chessman Image Dataset for chess piece classification
chess_dataset_path = "Chessman-image-dataset"
# Here, we assume that the images are stored in subdirectories named after the classes
classes = os.listdir(chess_dataset_path)
chess_dataset = []
for cls in classes:
    cls_path = os.path.join(chess_dataset_path, cls)
    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)
        img = cv2.imread(img_path)
        if img is not None and not img.size == 0:
            chess_dataset.append((img, cls))
            print(f"Added {img_path} to chess dataset with label {cls}")
        else:
            print(f"Failed to read {img_path}")

# Open the camera
cap = cv2.VideoCapture(0)

# Flag to indicate whether to continue showing images
continue_showing_images = True

# Loop through the frames from the camera
while continue_showing_images:
    ret, frame = cap.read()

    # Detect faces in the current frame
    faces = detect_faces(frame)

    # Draw rectangles around the detected faces and show a random chess piece image for each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if len(chess_dataset) > 0:
            img, cls = random.choice(chess_dataset)
            resized_img = cv2.resize(img, (w, h))
            frame[y:y+h, x:x+w] = resized_img

    # Display the current frame
    cv2.imshow('Face Detection', frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('u'):
        continue_showing_images = False
        last_frame = frame.copy()
    elif key == ord('q'):
        break

# Keep showing the last frame until the 'q' key is pressed
while True:
    cv2.imshow('Face Detection', last_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
