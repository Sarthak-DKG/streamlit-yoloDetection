import cv2
import streamlit as st
from ultralytics import YOLO
from utils import Utils
from utils import logo

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded"    # Expanding sidebar by default
)

logo()
st.header("Get Object Detection on Video using YOLOv8 Model: {modelName}".format(modelName = Utils.get_model_name()))
# Replace the relative path to your weight file
model_path = Utils.get_model() #"weights/yolov8n.pt"

# Creating sidebar
with st.sidebar:
    st.header("Image/Video Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting videos
    source_vid = st.sidebar.selectbox(
        "Choose a video...",
        ["Videos/cars.mp4", "Videos/Smoking_cigarrete.mp4"])

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("DKGLabs Object Detection using YOLOv8")

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
# st.write("Model loaded successfully!")

if source_vid is not None:
    print(str(source_vid))
    with open(str(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)
    if st.sidebar.button('Detect Objects'):
        vid_cap = cv2.VideoCapture( str(source_vid) )
        st_frame = st.empty()
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                image = cv2.resize(image, (720, int(720*(9/16))))
                res = model.predict(image, conf=confidence)
                result_tensor = res[0].boxes
                res_plotted = res[0].plot()
                st_frame.image(res_plotted,
                               caption='Detected Video',
                               channels="BGR",
                               use_column_width=True
                               )
            else:
                vid_cap.release()
                break