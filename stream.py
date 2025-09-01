import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import cv2

# ------------------------------
# Load TFLite model
# ------------------------------
tflite_model_path = "Models/25_08_float16.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------------------
# Preprocess image
# ------------------------------
def preprocess_image(img, input_shape=(640, 640)):
    img_resized = img.resize(input_shape)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array, np.array(img)

# ------------------------------
# Decode YOLO raw output
# ------------------------------
def decode_yolo_output(output, conf_threshold=0.25, input_size=640):
    output = np.squeeze(output)  # (5, 8400)
    boxes, scores = [], []

    for i in range(output.shape[1]):
        conf = output[4, i]
        if conf > conf_threshold:
            cx, cy, w, h = output[0, i], output[1, i], output[2, i], output[3, i]
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

            # Scale back to original image size
            x1, y1, x2, y2 = x1 * input_size, y1 * input_size, x2 * input_size, y2 * input_size
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)

    return np.array(boxes), np.array(scores)

# ------------------------------
# Non-Max Suppression
# ------------------------------
def non_max_suppression(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []

    boxes_cv2 = []
    for b in boxes:
        x1, y1, x2, y2 = b
        boxes_cv2.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

    indices = cv2.dnn.NMSBoxes(
        boxes_cv2, scores.tolist(),
        score_threshold=0.25,
        nms_threshold=iou_threshold
    )
    indices = np.array(indices).flatten() if len(indices) > 0 else []
    return indices

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🔑 Key Detection")
st.write("Upload an image to detect keys")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    # Preprocess and inference
    input_data, orig_img = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Decode + Apply NMS
    boxes, scores = decode_yolo_output(output_data, conf_threshold=0.30)
    indices = non_max_suppression(boxes, scores, iou_threshold=0.45)

    final_boxes = [boxes[i] for i in indices]
    final_scores = [scores[i] for i in indices]

    st.subheader(f"🎯 Keys detected: {len(final_boxes)}")

    for i, (box, score) in enumerate(zip(final_boxes, final_scores)):
        st.write(f"Key {i+1}: Accuracy = {score:.2%}")


