# for streamlit virtual env is key detection as init python = 3.13
# for label-studio virtual env is .venv as init python = 3.12

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import cv2
from label_studio_sdk.client import LabelStudio
import os
from dotenv import load_dotenv
import requests
from PIL import ImageFont
load_dotenv()


# ===== CONFIG =====
TFLITE_MODEL_PATH = "Models/25_08_float16.tflite"
LABEL_STUDIO_URL = "http://localhost:8080"
PROJECT_NAME = "Key Detection Project"
PROJECT_ID = 1          
STORAGE_ID = 7 
LOCAL_PATH = r"C:\Key Detection Model\images"
IMAGE_DIR = os.path.join(os.getcwd(), "images")
API_KEY = os.getenv("API_KEY")
# ===== Load TFLite model =====
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------
# Environment Checks
# -------------------------------
# set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
# echo %LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED%

# -------------------------------
# Powershell Checks
# -------------------------------
# $env:LOCAL_FILES_SERVING_ENABLED="true"
# $env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="C:\Key Detection Model\images"


# ===== Label Studio connection =====
ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

try:
    projects = ls.projects.list()
except Exception as e:
    st.error(f"❌ Cannot connect to Label Studio: {e}")
    st.stop()

# ===== Get or create project =====
project = None
for p in projects:
    if p.title == PROJECT_NAME:
        project = ls.projects.get(p.id)
        break

if not project:
    project = ls.projects.create(
        title=PROJECT_NAME,
        description="Label keys in images",
        label_config="""
        <View>
            <Image name="image" value="$image"/>
            <RectangleLabels name="label" toName="image">
                <Label value="Key"/>
            </RectangleLabels>
        </View>
        """
    )

# ===== Helper functions =====
def preprocess_image(img, input_shape=(640, 640)):
    orig_w, orig_h = img.size
    img_resized = img.resize(input_shape)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, (orig_w, orig_h)


def decode_yolo_output(output, conf_threshold=0.50, orig_size=(640, 640)):
    output = np.squeeze(output)
    boxes, scores = [], []
    orig_w, orig_h = orig_size
    if output.ndim != 2 or output.shape[0] < 5:
        return np.array(boxes), np.array(scores)
    for i in range(output.shape[1]):
        conf = output[4, i]
        if conf > conf_threshold:
            cx, cy, w, h = output[0, i], output[1, i], output[2, i], output[3, i]
            x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
            x1, y1, x2, y2 = int(x1*orig_w), int(y1*orig_h), int(x2*orig_w), int(y2*orig_h)
            boxes.append([x1, y1, x2, y2])
            scores.append(float(conf))
    return np.array(boxes), np.array(scores)


def non_max_suppression(boxes, scores, iou_threshold=0.45, score_threshold=0.50):
    if len(boxes) == 0:
        return []
    boxes_cv2 = [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for x1, y1, x2, y2 in boxes]
    indices = cv2.dnn.NMSBoxes(boxes_cv2, scores.tolist(),
                               score_threshold=score_threshold,
                               nms_threshold=iou_threshold)
    return np.array(indices).flatten() if len(indices) > 0 else []


def draw_boxes(image_pil, boxes, scores):
    draw = ImageDraw.Draw(image_pil)

    try:
        # Use a common font, fallback if not found
        font = ImageFont.truetype("arial.ttf", 80)  # 24px font size
    except:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), score in zip(boxes, scores):
        if score < 0.80:
            draw.rectangle([x1,y1,x2,y2], outline="yellow",width=15,)
        else:
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=15,)
        draw.text(
            (max(0, x1), max(0, y1 - 30)),  # move up a little
            f"{score:.2f}",
            fill="red",
            font=font
        )
    return image_pil


def get_access_token():
    """Exchange refresh token for short-lived access token."""
    resp = requests.post(
        f"{LABEL_STUDIO_URL}/api/token/refresh",
        headers={"Content-Type": "application/json"},
        json={"refresh": API_KEY}
    )
    if resp.ok:
        resp.json().get("access")
        return resp.json().get("access")
    else:
        raise RuntimeError(f"❌ Failed to refresh token: {resp.status_code} {resp.text}")


def upload_image_to_ls(file_path):
    access_token = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # headers = {"Authorization": f"Token {API_KEY}"}
    
    with open(file_path, "rb") as f:
        response = requests.post(
            f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/import",
            headers=headers,
            files={"file": f}
        )
   
    if response.ok:
        # Import succeeded, now fetch the last created task
        tasks_resp = requests.get(
            f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/tasks",
            headers=headers
        )
       
        if tasks_resp.ok:
            tasks = tasks_resp.json()
            if tasks and len(tasks) > 0:
                latest_task_id = tasks[0]["id"] 
                return latest_task_id
                
        else:
            st.error(f"❌ Could not fetch tasks: {tasks_resp.status_code} {tasks_resp.text}")
            return None
    else:
        st.error(f"❌ Upload failed: {response.status_code} {response.text}")
        return None


def send_predictions_to_ls(task_id, final_boxes, final_scores, orig_size):
    url = f"{LABEL_STUDIO_URL}/api/predictions/"
    access_token = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}

    orig_w, orig_h = orig_size
    results = []

    for box, score in zip(final_boxes, final_scores):
        x1, y1, x2, y2 = box
        results.append({
            "from_name": "label",        # matches config
            "to_name": "image",          # matches config
            "type": "rectanglelabels",
            "image_rotation": 0,
            "original_width": orig_w,
            "original_height": orig_h,
            "value": {
                "x": (x1 / orig_w) * 100,
                "y": (y1 / orig_h) * 100,
                "width": ((x2 - x1) / orig_w) * 100,
                "height": ((y2 - y1) / orig_h) * 100,
                "rotation": 0,
                "rectanglelabels": ["key"]  # ✅ must be lowercase
            },
             "score": float(score) 
        })

    payload = {
        "model_version": "v1",
        "task": task_id,
        "result": results,
        "score": float(max(final_scores)) if final_scores else 0.0
    }

    resp = requests.post(url, headers=headers, json=payload)
    # print("🔎 Prediction upload response:", resp.status_code, resp.text)

# -------------------------------
# Streamlit UI
# -------------------------------

st.title("🔑 Key Detection with HINL")
st.set_page_config(page_title="Key Counter App", layout="wide")
uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Model inference
    input_data, orig_size = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    boxes, scores = decode_yolo_output(output_data, conf_threshold=0.3, orig_size=orig_size)
    indices = non_max_suppression(boxes, scores)
    final_boxes = [boxes[i] for i in indices] if len(indices) else []
    final_scores = [scores[i] for i in indices] if len(indices) else []
    
    st.subheader(f"🎯 Keys detected: {len(final_boxes)}")
    for i, (box, score) in enumerate(zip(final_boxes, final_scores)):
        st.write(f"Key {i+1}: Accuracy = {score:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        pred_img = image.copy()
        pred_img = draw_boxes(pred_img, final_boxes, final_scores)
        st.image(pred_img, caption="Model Predictions", use_container_width=True)


# -------------------------------
# Annotate Button
# -------------------------------
if st.button("Annotate in Label Studio"):
    if uploaded_file is None:
        st.error("Please upload an image first!")
    else:
        # Save to local folder
        filename = uploaded_file.name
        file_path = os.path.join(IMAGE_DIR, filename)
        counter = 1
        base, ext = os.path.splitext(filename)
        while os.path.exists(file_path):
            filename = f"{base}_{counter}{ext}"
            file_path = os.path.join(IMAGE_DIR, filename)
            counter += 1
        image.save(file_path)

        # Upload directly to Label Studio
        task_id = upload_image_to_ls(file_path)
        if task_id:
            st.success("✅ Uploaded into Label Studio")
            st.markdown(f"[🔗 Open in Label Studio]({LABEL_STUDIO_URL}/projects/{PROJECT_ID}/data)")
            
            
            # # 👉 Now send bounding boxes as predictions
            # if final_boxes:
            #     send_predictions_to_ls(task_id, final_boxes, final_scores, orig_size)
            #     st.info("📤 Sent model predictions (bounding boxes) to Label Studio!")
            # 👉 If model detected any keys, send predictions to LS
            
            
            if final_boxes:
                send_predictions_to_ls(task_id, final_boxes, final_scores, orig_size)
                st.info(
                    f"📤 Sent {len(final_boxes)} model predictions (bounding boxes) to Label Studio!"
                )
            else:
                # Send an empty prediction to indicate model ran but found nothing
                send_predictions_to_ls(task_id, [], [], orig_size)
                st.warning(
                    "⚠️ Model did not detect any keys in this image. You can manually annotate in Label Studio."
                )
            st.markdown("---")
            st.subheader("📑 Label Studio Interface (Embedded)")

            # Embed LS into the Streamlit app
            LABEL_STUDIO_URL_Custom = "http://localhost:8080/projects/1/data"
            st.components.v1.iframe(f"{LABEL_STUDIO_URL_Custom}", height=1200, scrolling=True)
        else:
            st.error("❌ Failed to upload the image to Label Studio. Please check your connection or API key.")
            
# -------------------------------
# Embed Label Studio UI
# -------------------------------

