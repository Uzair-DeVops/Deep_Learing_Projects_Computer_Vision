import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

# Load two models
model1 = YOLO("model.pt")  # Replace with your actual watch detection model
model2 = YOLO("yolov8n.pt")      # General model or another custom one

# App title
st.title("📸 Dual YOLO Model Detection")

# Sidebar for mode
mode = st.sidebar.selectbox("Choose Mode", ["Webcam Detection", "Image Upload"])

# Session state for webcam
if 'running' not in st.session_state:
    st.session_state.running = False

# ======================
# Webcam Mode
# ======================
if mode == "Webcam Detection":
    st.subheader("🎥 Webcam Detection with Two Models")

    # Start and Stop Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Webcam"):
            st.session_state.running = True
    with col2:
        if st.button("⛔ Stop Webcam"):
            st.session_state.running = False

    # Image display area
    frame_placeholder = st.empty()

    def run_webcam():
        cap = cv2.VideoCapture(0)
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam not working.")
                break

            # Run both models
            results1 = model1(frame)
            results2 = model2(frame)

            # Get annotated frames
            frame1 = results1[0].plot()
            frame2 = results2[0].plot()

            # Combine both annotations (add them like layers)
            combined = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)

            # Convert for Streamlit
            combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(combined, channels="RGB", use_container_width=True)

            time.sleep(0.03)

        cap.release()

    if st.session_state.running:
        run_webcam()

# ======================
# Image Upload Mode
# ======================
elif mode == "Image Upload":
    st.subheader("🖼️ Upload an Image for Detection (Both Models)")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        # Run both models
        res1 = model1(img_np)
        res2 = model2(img_np)

        img1 = res1[0].plot()
        img2 = res2[0].plot()

        # Merge both detections
        final = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
        final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

        st.image(final, caption="Combined Detection", use_container_width=True)


        # Display individual results