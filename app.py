# import numpy as np
# import base64
# from io import BytesIO
# import cv2
# from PIL import Image
# from flask import Flask, request, jsonify
#
# app = Flask(__name__)
#
# def crop_and_resize_face(image_data, face_scale_percentage=40, head_to_height_ratio=0.5, shift_x=0, shift_y=0,
#                          max_width=113, max_height=132, dpi=(144, 144)):
#     """
#     Crop and resize face from base64 image data based on the provided parameters and return processed image as base64.
#
#     Parameters:
#     - image_data: Binary image data from decoded base64 string
#     - face_scale_percentage: Percentage (10 to 500) to scale the face region
#     - head_to_height_ratio: Ratio of the head size to the total height in the cropped image
#     - shift_x: Horizontal shift of the face region
#     - shift_y: Vertical shift of the face region
#     - max_width: Maximum width of the output image
#     - max_height: Maximum height of the output image
#     - dpi: Desired DPI of the output image
#
#     Returns:
#     - Processed image as base64 string
#     """
#     face_scale_percentage = max(10, min(face_scale_percentage, 500))
#     face_scale = face_scale_percentage / 100.0
#
#     # Decode the image data and load it into OpenCV
#     np_arr = np.frombuffer(image_data, np.uint8)
#     image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#
#     if image is None:
#         return None, "Unable to decode image."
#
#     original_height, original_width = image.shape[:2]
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
#
#     if len(faces) == 0:
#         return None, "No faces detected."
#
#     (x, y, w, h) = faces[0]
#     total_height = int(h / head_to_height_ratio)
#
#     scaled_w = int(w / face_scale)
#     scaled_h = int(total_height / face_scale)
#     aspect_ratio = original_width / original_height
#     scaled_h = int(scaled_w / aspect_ratio)
#
#     x_center = x + w // 2
#     y_center = y + h // 2
#     x_new = max(0, x_center - scaled_w // 2 + shift_x)
#     y_new = max(0, y_center - scaled_h // 2 + shift_y)
#
#     if x_new + scaled_w > original_width:
#         scaled_w = original_width - x_new
#     if y_new + scaled_h > original_height:
#         scaled_h = original_height - y_new
#
#     face_crop = image[y_new:y_new + scaled_h, x_new:x_new + scaled_w]
#     face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
#     pil_image = Image.fromarray(face_crop_rgb)
#
#     if pil_image.width > max_width or pil_image.height > max_height:
#         pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
#
#     pil_image = pil_image.convert("RGBA")
#
#     buffered = BytesIO()
#     pil_image.save(buffered, format="PNG", dpi=dpi)
#     buffered.seek(0)
#     processed_image_data = buffered.getvalue()
#
#     processed_base64_string = base64.b64encode(processed_image_data).decode('utf-8')
#     return processed_base64_string, None
#
#
# @app.route('/process-image', methods=['POST'])
# def process_image():
#     data = request.json
#     base64_string = data.get('image')
#
#     # Grab configuration parameters from the request body, with default values if not provided
#     face_scale_percentage = data.get('face_scale_percentage', 40)
#     head_to_height_ratio = data.get('head_to_height_ratio', 0.5)
#     shift_x = data.get('shift_x', 0)
#     shift_y = data.get('shift_y', 0)
#     max_width = data.get('max_width', 113)
#     max_height = data.get('max_height', 132)
#     dpi_x = data.get('dpi_x', 144)
#     dpi_y = data.get('dpi_y', 144)
#
#     if not base64_string:
#         return jsonify({"error": "No image provided"}), 400
#
#     if base64_string.startswith('data:image/png;base64,'):
#         base64_string = base64_string.split(',')[1]
#
#     try:
#         image_data = base64.b64decode(base64_string)
#     except Exception as e:
#         return jsonify({"error": "Invalid base64 string"}), 400
#
#     # Pass the parameters to the crop_and_resize_face function
#     processed_image_base64, error = crop_and_resize_face(
#         image_data,
#         face_scale_percentage=face_scale_percentage,
#         head_to_height_ratio=head_to_height_ratio,
#         shift_x=shift_x,
#         shift_y=shift_y,
#         max_width=max_width,
#         max_height=max_height,
#         dpi=(dpi_x, dpi_y)
#     )
#
#     if error:
#         return jsonify({"error": error}), 400
#
#     return jsonify({'processed_image': processed_image_base64})
#
#
# if __name__ == '__main__':
#     app.run(debug=True)


import base64
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import io

app = Flask(__name__)

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
