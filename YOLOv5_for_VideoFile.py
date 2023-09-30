# import streamlit as st
import cv2
from yolo_predictions import YOLO_Pred

vidcap = cv2.VideoCapture(".//Videos/cars.mp4") # For Video

yolo = YOLO_Pred('./models/generic_YOLOv5/best.onnx',
                 './models/generic_YOLOv5/data.yaml')

while True:
    success,img = vidcap.read()
    # pred_img = img
    pred_img = yolo.predictions(img)
    cv2.imshow("Image",pred_img)
    if cv2.waitKey(1) == ord('q'):
        break

vidcap.release()    
cv2.destroyAllwindows()