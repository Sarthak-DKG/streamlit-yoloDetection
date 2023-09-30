import streamlit as st 
from streamlit_webrtc import webrtc_streamer
import av
import ultralytics
from utils import Utils
from utils import logo
from ultralytics import YOLO
import cv2
import cvzone
import math
from utils import logo
import yaml

logo()
st.header("Get Object Detection on LiveCam using YOLOv8 Model: {modelName}".format(modelName = Utils.get_model_name()))
# load yolo model
model = YOLO(Utils.get_model())
names = []
with open(Utils.get_yaml(), 'r') as file11:
        names = yaml.safe_load(file11)["names"]

print("@@@@@@")
print(Utils.get_model())
print(Utils.get_yaml())
print("@@@@@")


def video_frame_callback(frame):
    cap = frame.to_ndarray(format="bgr24")
    #any operation
    #flipped = img[::-1,:,:]
    results = model(cap,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1),int(x2),int(y2)

            w,h = x2-x1,y2-y1
            cvzone.cornerRect(cap, (x1,y1,w,h))

            conf = math.ceil((box.conf[0]*100))/100

            cls = int(box.cls[0])

            cvzone.putTextRect(cap,f'{names[cls]} {conf}', (max(0,x1), max(35,y1)), scale=1, thickness=1)

    
    return av.VideoFrame.from_ndarray(cap, format="bgr24")


webrtc_streamer(key="example", 
                     video_frame_callback=video_frame_callback,
                     media_stream_constraints={"video":True, "audio":False})