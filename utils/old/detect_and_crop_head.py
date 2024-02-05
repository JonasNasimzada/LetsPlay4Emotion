import cv2
import numpy as np
from PIL import Image

#sys.path.append("./stargan-v2/core")
#import wing


def detect_and_crop_head_and_interpolate(args, input_image, output_image_path, factor=1.7):
    # Load the pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert PIL image to OpenCV format (BGR)
    cv_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        # Assuming the first face is the target, you can modify this based on your requirements
        x, y, w, h = faces[0]

        # Calculate the new coordinates and dimensions for a 1:1 aspect ratio
        center_x = x + w // 2
        center_y = y + h // 2
        size = int(max(w, h) * factor)
        x_new = max(0, center_x - size // 2)
        y_new = max(0, center_y - size // 2)

        # Crop the head region with a 1:1 aspect ratio
        cropped_head = cv_image[y_new:y_new + size, x_new:x_new + size]

        # Convert the cropped head back to PIL format
        cropped_head_pil = Image.fromarray(cv2.cvtColor(cropped_head, cv2.COLOR_BGR2RGB))

        wing.align_faces(args, cropped_head_pil, output_image_path)
        # Save the cropped head image
        # cropped_head_pil.save(output_image_path)
        print("Cropped head saved successfully.")
    else:
        print("No faces detected in the input image.")


def detect_and_crop_head_video(input_image, factor=1):
    # Load the pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert PIL image to OpenCV format (BGR)
    cv_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        # Assuming the first face is the target, you can modify this based on your requirements
        x, y, w, h = faces[0]

        # Calculate the new coordinates and dimensions for a 1:1 aspect ratio
        center_x = x + w // 2
        center_y = y + h // 2
        size = int(max(w, h) * factor)
        x_new = max(0, center_x - size // 2)
        y_new = max(0, center_y - size // 2)

        # Crop the head region with a 1:1 aspect ratio
        cropped_head = cv_image[y_new:y_new + size, x_new:x_new + size]

        # Convert the cropped head back to PIL format
        return cropped_head
