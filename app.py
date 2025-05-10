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
    """Apply Contrast Limited Adaptive Histogram Equalization to enhance image contrast."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

@st.cache_resource
def load_model(model_path: str) -> YOLO:
    """Load and cache a YOLO model."""
    return YOLO(model_path)

def detect_defects(frame: np.ndarray, bottle_model: YOLO, defect_model: YOLO, 
                  conf_threshold: float = 0.5) -> Tuple[np.ndarray, List[Dict]]:
    """
    Two-stage detection: first detect bottles, then detect defects within each bottle.
    
    Args:
        frame: Input image
        bottle_model: YOLO model for bottle detection
        defect_model: YOLO model for defect detection
        conf_threshold: Confidence threshold for detections
        
    Returns:
        Annotated frame and list of detected defects
    """
    annotated_frame = frame.copy()
    detected_defects = []
    
    # First stage: Detect bottles
    first_stage_results = bottle_model(frame)[0]
    bottles = first_stage_results.boxes
    
    # Process each detected bottle
    for bottle in bottles:
        bottle_box = bottle.xyxy.cpu().numpy()[0]
        bottle_score = float(bottle.conf.cpu().numpy()[0])
        bottle_class = int(bottle.cls.cpu().numpy()[0])
        
        # Skip if confidence is low or not a bottle
        if bottle_score < conf_threshold or bottle_class != 0:
            continue
            
        # Extract bottle coordinates and draw bounding box
        x1, y1, x2, y2 = map(int, bottle_box)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Crop the bottle region
        cropped_img = frame[y1:y2, x1:x2]
        if cropped_img.size == 0:
            continue
            
        # Apply CLAHE to enhance contrast in the cropped image
        enhanced_img = apply_clahe(cropped_img)
        
        # Second stage: Detect defects within the bottle region
        # Use the enhanced image directly without preprocessing
        defect_results = defect_model(enhanced_img)[0]
        
        # Process each detected defect
        for defect in defect_results.boxes:
            defect_box = defect.xyxy.cpu().numpy()[0]
            defect_score = float(defect.conf.cpu().numpy()[0])
            defect_class = int(defect.cls.cpu().numpy()[0])
            
            # Skip if confidence is low
            if defect_score < conf_threshold:
                continue
                
            # Get defect coordinates within the cropped image
            rx1, ry1, rx2, ry2 = map(int, defect_box)
            
            # Make sure coordinates are within the cropped image bounds
            h, w = cropped_img.shape[:2]
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(w, rx2), min(h, ry2)
            
            # Calculate absolute coordinates in the original image
            abs_x1 = x1 + rx1
            abs_y1 = y1 + ry1
            abs_x2 = x1 + rx2
            abs_y2 = y1 + ry2
            
            # Get defect label and format display text
            label = defect_model.names[defect_class]
            label_with_score = f"{label}: {defect_score:.2f}"
            
            # Store defect information
            detected_defects.append({
                "label": label,
                "confidence": defect_score,
                "coordinates": [abs_x1, abs_y1, abs_x2, abs_y2]
            })
            
            # Draw defect bounding box and label on the annotated frame
            cv2.rectangle(annotated_frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
            cv2.putText(annotated_frame, label_with_score, (abs_x1, abs_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
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
            
            # Display original image
            st.subheader("Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            # Process image and display results
            result_img, defects = detect_defects(image, bottle_model, defect_model, conf_threshold)
            st.subheader("Detected Defects")
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            # Display defect summary
            if defects:
                st.subheader("Defect Summary")
                for i, defect in enumerate(defects, 1):
                    st.write(f"{i}. **{defect['label']}** with {defect['confidence']:.2f} confidence")
            else:
                st.info("No defects detected with the current confidence threshold.")

    else:  # Video upload
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            
            # Process video
            st.subheader("Video Analysis")
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            defect_counts = {}
            frame_count = 0
            
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                annotated_frame, defects = detect_defects(frame, bottle_model, defect_model, conf_threshold)
                
                # Update defect counts
                for d in defects:
                    defect_counts[d['label']] = defect_counts.get(d['label'], 0) + 1
                
                # Display current frame
                stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                
                # Update progress
                frame_count += 1
                progress_bar.progress(min(frame_count / total_frames, 1.0))
                
            cap.release()
            os.unlink(tfile.name)

            # Display defect summary for video
            st.subheader("Video Defect Summary")
            if defect_counts:
                for label, count in defect_counts.items():
                    st.write(f"- **{label}**: {count} occurrences")
            else:
                st.info("No defects detected with the current confidence threshold.")

if __name__ == "__main__":
    main()
