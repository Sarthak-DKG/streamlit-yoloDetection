
import streamlit as st 

from streamlit_webrtc import webrtc_streamer
import av
from yolo_predictions import YOLO_Pred
from utils import Utils
from utils import logo

logo()
st.header("Get Object Detection on LiveCam using YOLOv5 Model: {modelName}".format(modelName = Utils.get_model_name()))

# load yolo model
yolo = YOLO_Pred(onnx_model=Utils.get_model(),
                    data_yaml=Utils.get_yaml())

print("@@@@@@")
print(Utils.get_model())
print(Utils.get_yaml())
print(Utils.get_model_name())
print("@@@@@")


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    #any operation
    #flipped = img[::-1,:,:]
    pred_img = yolo.predictions(img)
    
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="example", 
                     video_frame_callback=video_frame_callback,
                     media_stream_constraints={"video":True, "audio":False})