import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
import cv2
import streamlit as st
import numpy as np
import sys
from tempfile import NamedTemporaryFile

# Add the inner ultralytics folder to sys.path
sys.path.insert(0, os.path.abspath('ultralytics'))
from ultralytics import YOLO

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

st.set_page_config(page_title="Bottle Defects Detection", layout="wide")
st.title("Bottle Defects Detection")

# Load models
first_model = load_model('best.pt')
re_detection_model = load_model('defect_detection.pt')

file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

def process_frame(frame):
    # First detection
    first_stage_results = first_model(frame)[0]
    annotated_frame = frame.copy()

    boxes = first_stage_results.boxes.xyxy.cpu().numpy()
    scores = first_stage_results.boxes.conf.cpu().numpy()
    classes = first_stage_results.boxes.cls.cpu().numpy()

    for box, score, class_id in zip(boxes, scores, classes):
        if score < 0.5 or int(class_id) != 0:
            continue
        x1, y1, x2, y2 = map(int, box)
        cropped_img = frame[y1:y2, x1:x2]
        if cropped_img.size == 0:
            continue

        re_detection_results = re_detection_model(cropped_img)[0]

        for r_box, r_score, r_class_id in zip(
            re_detection_results.boxes.xyxy.cpu().numpy(),
            re_detection_results.boxes.conf.cpu().numpy(),
            re_detection_results.boxes.cls.cpu().numpy()
        ):
            if r_score < 0.5:
                continue
            rx1, ry1, rx2, ry2 = map(int, r_box)
            abs_x1 = x1 + rx1
            abs_y1 = y1 + ry1
            abs_x2 = x1 + rx2
            abs_y2 = y1 + ry2

            label = re_detection_model.names[int(r_class_id)]
            label_with_score = f"{label}: {r_score:.2f}"

            cv2.rectangle(annotated_frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
            cv2.putText(annotated_frame, label_with_score, (abs_x1, abs_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

if file:
    if file.type.startswith('image'):
        # Image case
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        annotated_img = process_frame(img)
        st.image(annotated_img, channels="RGB", caption="Detected Image")
    elif file.type == "video/mp4":
        # Video case
        tfile = NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame = process_frame(frame)
            stframe.image(annotated_frame, channels="RGB")
        cap.release()
