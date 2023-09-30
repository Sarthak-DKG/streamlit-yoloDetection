# import streamlit as st
import cv2
from yolo_predictions import YOLO_Pred
import ultralytics
from ultralytics import YOLO
import cvzone
import math
import yaml
import sys


if len(sys.argv) == 4:
    names = []
    with open(sys.argv[2], 'r') as file11:
            names = yaml.safe_load(file11)["names"]

    vidcap = cv2.VideoCapture(sys.argv[3])
    model = YOLO(sys.argv[1])
else :
    vidcap = cv2.VideoCapture(".//Videos/Smoking_cigarrete.mp4")
    model = YOLO("models/Cigarrets_YOLOv8/best.pt")
    names = ["Cigarette"]

fps = int(vidcap.get(cv2.CAP_PROP_FPS))
print(fps, " fps")
fps = fps / 5
frame_count = 0
while True:    
    success, img = vidcap.read()
    if frame_count % fps == 0:
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # cvzone.putTextRect(img, f'{conf}', (max(0,x1) ,max(35,y1)))

                # ClassName
                cls = int(box.cls[0])
                if conf > 0.67:
                    cvzone.putTextRect(
                        img,
                        f"{names[cls]} {conf}",
                        (max(0, x1), max(35, y1)),
                        scale=1,
                        thickness=1,
                    )
        cv2.imshow("Video", img)
        if cv2.waitKey(1) == ord('q'):
            break;
    frame_count += 1

vidcap.release()
cv2.destroyAllwindows()