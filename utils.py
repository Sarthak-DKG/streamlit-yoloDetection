import yaml
import os
from streamlit_extras.app_logo import add_logo
import streamlit as st

def logo():
    add_logo("./images/dkg_logo.png", height=300)

def load_folders(root_folder):
    folders = []
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            folders.append(folder_name)
    return folders

# Function to load filenames in each folder into a dictionary
def load_filenames(root_folder):
    folder_files = {}
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            folder_files[folder_name] = []
            for file_name in os.listdir(folder_path):
                folder_files[folder_name].append(file_name)
    return folder_files


class Utils:
    # def __init__(self) -> None:
    #     pass
    model_name = ""
    onnx_model = ""
    data_yaml = ""
    model_root = ""
    model_folders = []
    model_folder_files = {}

    @classmethod 
    def load_model_data(self,istr):
        Utils.model_root = istr
        Utils.model_folders = load_folders(istr)
        Utils.model_folder_files = load_filenames(istr)
        print(Utils.model_folders)
        print(Utils.model_folder_files)
        return Utils.model_folders, Utils.model_folder_files 

    @classmethod 
    @st.cache_data 
    def set_model(self,istr):
        Utils.model_name = istr
        modelFileName = "/best.onnx" if istr[-2:] == "v5" else  "/best.pt"
        Utils.onnx_model = istr if istr == "" else (Utils.model_root + "/" + istr + modelFileName)

    @classmethod
    def set_yaml(self,istr):
         Utils.data_yaml = istr if istr == "" else (Utils.model_root + "/" + istr + "/data.yaml")
         
    @classmethod 
    def get_model_name(self):
        return Utils.model_name
         
    @classmethod 
    def get_model(self):
        return Utils.onnx_model

    @classmethod
    def get_yaml(self):
        return Utils.data_yaml
    
    @classmethod
    def print_model(self):
        print(Utils.onnx_model,"*", Utils.data_yaml)
