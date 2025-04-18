import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Load two models
model1 = YOLO("model.pt")  # Replace with your actual watch detection model
model2 = YOLO("yolov8n.pt")  # General model or another custom one

# App title
st.title("📸 Dual YOLO Model Detection")

# Sidebar for mode
mode = st.sidebar.selectbox("Choose Mode", ["Webcam Detection", "Image Upload"])

# ======================
# Webcam Mode (WebRTC Solution)
# ======================

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model1 = model1
        self.model2 = model2

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the frame to a numpy array
        img = frame.to_ndarray(format="bgr24")

        # Run both models
        results1 = self.model1(img)
        results2 = self.model2(img)

        # Get annotated frames
        frame1 = results1[0].plot()
        frame2 = results2[0].plot()

        # Combine both annotations (add them like layers)
        combined = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)

        # Convert back to VideoFrame
        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        return av.VideoFrame.from_ndarray(combined, format="rgb24")

if mode == "Webcam Detection":
    st.subheader("🎥 Webcam Detection with Two Models")

    # Streamlit-webrtc: Start webcam stream
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

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
        st.success("Detection complete!")
