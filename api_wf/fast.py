from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from paittern.keypoints.keypoint_model import run_model_gif
from paittern.keypoints.video_gif import video_to_gif
from paittern.selection.selection_pose import get_best_poses
import os
from dotenv import load_dotenv, find_dotenv
import requests
from tensorflow.keras.utils import CustomObjectScope
import tensorflow as tf
from paittern.contouring.predict import predict_mask



'''API to obtain inputs from user (video, height and pattern desired) and predict
body measurments given our packages- this will be send to the pattern app which
will provide the desired pattern and return it on streamlit'''

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    print('hey')
    return {"greeting": "Hello world"}


@app.get("/predict")
def predict(url,pattern,height=168):

    # first step is to retrieve the video_file : streamlit file will save the video
    #into a cloud storage and return as a parameter the url of video's path
    # so first step is to get the video from the url provided

    gif = video_to_gif(url)
    print('flag1 - gif done')
    print(gif)
    keypoints_sequence, output_images= run_model_gif(gif)
    print('flag2 - run done')

    best_poses_idx=  get_best_poses(keypoints_sequence,output_images)
    print('flag3 - run done')
    with CustomObjectScope():
        model = tf.keras.models.load_model("./paittern/contouring/contouring_model")
    print(model)
    #countouring
    print('flag4 - enter contour')
    mask_list=[]
    for idx in best_poses_idx:
        mask = predict_mask(output_images[idx],model)
        mask_list.append(mask)
    print(mask_list)





    return {'working' :  'SUCCESS','mask_list':mask_list, 'best poses':best_poses_idx,'youy':'youy'}
