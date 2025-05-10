import cv2
import streamlit as st
import numpy as np
import sys
import os
from typing import Tuple, List, Dict
import tempfile

# Add the inner ultralytics folder to sys.path
sys.path.insert(0, os.path.abspath('ultralytics'))
from ultralytics import YOLO

os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def preprocess(img, target_size=640) -> Tuple[np.ndarray, float, int, int]:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return padded, scale, new_w, new_h

@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)

def detect_defects(frame: np.ndarray, bottle_model: YOLO, defect_model: YOLO, 
                  conf_threshold: float = 0.5) -> Tuple[np.ndarray, List[Dict]]:
    annotated_frame = frame.copy()
    detected_defects = []
    first_stage_results = bottle_model(frame)[0]
    boxes = first_stage_results.boxes.xyxy.cpu().numpy()
    scores = first_stage_results.boxes.conf.cpu().numpy()
    classes = first_stage_results.boxes.cls.cpu().numpy()

    for box, score, class_id in zip(boxes, scores, classes):
        if score < conf_threshold or int(class_id) != 0:
            continue
        x1, y1, x2, y2 = map(int, box)

        # Add padding
        pad = 20
        height, width = frame.shape[:2]
        pad_x1 = max(x1 - pad, 0)
        pad_y1 = max(y1 - pad, 0)
        pad_x2 = min(x2 + pad, width)
        pad_y2 = min(y2 + pad, height)

        # Crop and preprocess
        cropped_img = frame[pad_y1:pad_y2, pad_x1:pad_x2]
        cropped_img = apply_clahe(cropped_img)
        if cropped_img.size == 0:
            continue

        processed_img, scale_factor, new_w, new_h = preprocess(cropped_img)

        # Run defect detection on the CLAHE image (use cropped_img, not processed_img)
        defect_results = defect_model(cropped_img)[0]
        for r_box, r_score, r_class_id in zip(
            defect_results.boxes.xyxy.cpu().numpy(),
            defect_results.boxes.conf.cpu().numpy(),
            defect_results.boxes.cls.cpu().numpy()
        ):
            if r_score < conf_threshold:
                continue
            rx1, ry1, rx2, ry2 = map(int, r_box)

            # Clip to image size
            crop_h, crop_w = cropped_img.shape[:2]
            rx1, ry1 = min(rx1, crop_w), min(ry1, crop_h)
            rx2, ry2 = min(rx2, crop_w), min(ry2, crop_h)

            # Convert to original frame coordinates
            abs_x1 = pad_x1 + rx1
            abs_y1 = pad_y1 + ry1
            abs_x2 = pad_x1 + rx2
            abs_y2 = pad_y1 + ry2

            label = defect_model.names[int(r_class_id)]
            label_with_score = f"{label}: {r_score:.2f}"
            detected_defects.append({
                "label": label,
                "confidence": float(r_score),
                "coordinates": [abs_x1, abs_y1, abs_x2, abs_y2]
            })
            cv2.rectangle(annotated_frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
            cv2.putText(annotated_frame, label_with_score, (abs_x1, abs_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Optionally draw the bottle bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return annotated_frame, detected_defects


def main():
    st.set_page_config(page_title="Bottle Defects Detection", layout="wide")
    st.title("🧪 Bottle Defects Detection System")
    st.markdown("Upload an image or video for analysis using a two-stage YOLO-based detector.")

    # Sidebar settings
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    # Load models
    with st.spinner("Loading models..."):
        bottle_model = load_model('best32.pt')
        defect_model = load_model('best16.pt')

    upload_option = st.radio("Upload Type", ("Image", "Video"))

    if upload_option == "Image":
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            result_img, defects = detect_defects(image, bottle_model, defect_model, conf_threshold)
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width = True)
            st.subheader("Defect Summary")
            for defect in defects:
                st.write(f"- **{defect['label']}** at {defect['coordinates']} with {defect['confidence']:.2f} confidence")

    else:  # Video upload
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            defect_counts = {}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                annotated_frame, defects = detect_defects(frame, bottle_model, defect_model, conf_threshold)
                for d in defects:
                    defect_counts[d['label']] = defect_counts.get(d['label'], 0) + 1
                stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            cap.release()

            st.subheader("Video Defect Summary")
            for label, count in defect_counts.items():
                st.write(f"- **{label}**: {count}")

if __name__ == "__main__":
    main()
