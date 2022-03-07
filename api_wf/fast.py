from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from paittern.keypoints.keypoint_model import run_model_gif
from paittern.keypoints.video_gif import video_to_gif
from paittern.selection.selection_pose import get_best_poses
import os
from dotenv import load_dotenv, find_dotenv
import requests


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
    return {"greeting": "Hello world"}


@app.get("/predict")
def predict(url_video_path,pattern,height=168):



    # first step is to retrieve the video_file : streamlit file will save the video
    #into a cloud storage and return as a parameter the url of video's path
    # so first step is to get the video from the url provided

    gif = video_to_gif(url_video_path)
    keypoints_sequence, output_images= run_model_gif(gif)
    best_poses_idx=  get_best_poses(keypoints_sequence,output_images)

    for img in best_poses_idx:
        contour = get_contour(gif[img]) # a modifier, constitue la réalisation des contours de l'image
        #draw_prediction_on_image(contour,keypoints_sequence[img])

    measures_pred = get_mensuration(best_poses_idx,pattern,height) # fonction romain


    # translate height to desired unit
    # formating pattern type



    return {'Am I working?' :  'YESSSS'}
