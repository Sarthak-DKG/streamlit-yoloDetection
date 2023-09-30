import streamlit as st 
import yaml
from utils import Utils
from utils import logo

print("PageName")
print(__name__)

st.set_page_config(page_title="Home",
                   layout='wide',
                   page_icon='./images/home.png')   

st.title("YOLO Object Detection App")
st.caption('This web application demostrate Object Detection')
st.header("Which model would you like to use?")
logo()
folderList, fileDict = Utils.load_model_data("./models")
Utils.set_model("")
Utils.set_yaml("")

print("@".join(folderList))
model_selected = st.selectbox("Which model would you like to use?"  ,folderList , index=None, placeholder="Choose your model")

print(model_selected)
if model_selected != None:
    Utils.set_model(model_selected) 
    Utils.set_yaml(model_selected)
    if model_selected[-2:] == "v5":
        st.markdown("""
            # - [Image Detection](/YOLO_for_image/)
            # - [Live Camera Detection](/YOLO_for_LiveCam/)   
                        """)
        st.write("This model helps in detecting using Yolo V5")
    else:
        st.markdown("""
            # - [Image Detection](/YOLOv8_for_image/)
            # - [Live Camera Detection](/YOLOv8_for_LiveCam/) 
            # - [Video Detection](/YOLOv8_for_Video/)
                        """)  
        st.write("This model helps in detecting using Yolo V8")


    with open(Utils.get_yaml(), 'r') as file1:
        yaml_file1_data = yaml.safe_load(file1)
        st.write(yaml_file1_data['names'])
