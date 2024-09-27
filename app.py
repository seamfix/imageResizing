import base64
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io

app = Flask(__name__)

# Apply CORS to the app (this line ensures the CORS import is used)
CORS(app)

# Your existing function to process the face
def DrawFace(image, face_rect, scale_factor, shift_x, shift_y, window_width, window_height):
    x, y, w, h = face_rect
    scaled_w = int(w * scale_factor)
    scaled_h = int(h * scale_factor)

    # Adjust center based on the shift values
    x_center = x + w // 2 + shift_x
    y_center = y + h // 2 + shift_y

    # Calculate new crop coordinates
    x_new = max(0, x_center - scaled_w // 2)
    y_new = max(0, y_center - scaled_h // 2)

    # Make sure we don't go out of image bounds
    x_new = min(x_new, image.shape[1] - scaled_w)
    y_new = min(y_new, image.shape[0] - scaled_h)

    # Crop the image
    cropped_face = image[y_new:y_new + scaled_h, x_new:x_new + scaled_w]

    # Calculate the new aspect ratio and resize
    original_aspect_ratio = cropped_face.shape[1] / cropped_face.shape[0]
    target_aspect_ratio = window_width / window_height

    if original_aspect_ratio > target_aspect_ratio:
        new_width = window_width
        new_height = int(window_width / original_aspect_ratio)
    else:
        new_height = window_height
        new_width = int(window_height * original_aspect_ratio)

    # Resize the cropped face to the target dimensions
    face_resized = cv2.resize(cropped_face, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    return face_resized


def crop_and_resize_face(image_data, face_scale_percentage=100, shift_x=0, shift_y=0, window_width=300,
                         window_height=350):
    scale_factor = face_scale_percentage / 100.0
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        raise ValueError("No faces detected")

    face_rect = faces[0]
    cropped_face = DrawFace(image, face_rect, scale_factor, shift_x, shift_y, window_width, window_height)

    # Convert the image to PIL format for further processing
    cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cropped_face_rgb)

    return pil_image


# Flask route remains the same

# Flask route
@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        # Get the base64-encoded image from the request
        data = request.get_json()
        image_base64 = data['image_base64']

        # Decode base64 to image data
        image_data = base64.b64decode(image_base64)

        # Process the image
        processed_image = crop_and_resize_face(image_data, face_scale_percentage=190, shift_x=10, shift_y=-5)

        # Save processed image to bytes
        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG")
        processed_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({'processed_image_base64': processed_image_base64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
