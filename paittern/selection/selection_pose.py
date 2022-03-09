import numpy as np
import tensorflow as tf
from tensorflow.io import read_file
from tensorflow.image import decode_gif
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from paittern import keypoints
from paittern.keypoints import *
from paittern.keypoints.keypoint_model import run_model_gif
from paittern.keypoints.video_gif import video_to_gif
import cv2

#variables

#n= number of index to save from criteria 1
def get_best_poses(keypoints_sequence,images,n=10):
    print('Enter best_pose model')
    num_frames = np.array(keypoints_sequence).shape[0]
    criteria = np.empty((num_frames,4))
    best_poses_idx=[]
    for frame_idx in range(num_frames): # loop sur chaque image
        '''create a criteria matrix to iterate over '''
        criteria[frame_idx][0]=keypoints_sequence[frame_idx][0][0][5,1]-keypoints_sequence[frame_idx][0][0][6,1] # distance btwn shoulders
        criteria[frame_idx][1]=np.std([keypoints_sequence[frame_idx][0][0][5][0],
                                     keypoints_sequence[frame_idx][0][0][6][0],
                                     keypoints_sequence[frame_idx][0][0][7][0],
                                     keypoints_sequence[frame_idx][0][0][8][0],
                                     keypoints_sequence[frame_idx][0][0][9][0],
                                     keypoints_sequence[frame_idx][0][0][10][0]]) # std y's for arms alignement
        criteria[frame_idx][2]=keypoints_sequence[frame_idx][0][0][9,0]+keypoints_sequence[frame_idx][0][0][10,0] #sum of y's wrist for arms along the body
        criteria[frame_idx][3]=np.std([keypoints_sequence[frame_idx][0][0][5][1],\
                                      keypoints_sequence[frame_idx][0][0][6][1],keypoints_sequence[frame_idx][0][0][11][1],\
                                      keypoints_sequence[frame_idx][0][0][12][1],keypoints_sequence[frame_idx][0][0][13][1],\
                                      keypoints_sequence[frame_idx][0][0][14][1],keypoints_sequence[frame_idx][0][0][15][1],\
                                      keypoints_sequence[frame_idx][0][0][16][1]]) #std of x's for profile
    #selection pose face
    idx = (-criteria[:,0]).argsort()[:n] # select best 10 frames with max distance
    #selection best pose bras en croix
    min_std = min(criteria[:,1][idx])
    idx_face = np.where(criteria[:,1]==min_std)[0]
    best_poses_idx.append(idx_face[0])

    #pose bras long du corps
    max_arm_down = max(criteria[:,2][idx])
    idx_face2 = np.where(criteria[:,2]==max_arm_down)[0]
    best_poses_idx.append(idx_face2[0])

    #pose profil

    min_std_profile = min(criteria[:,3])
    idx_face = np.where(criteria[:,3]==min_std_profile)[0]
    best_poses_idx.append(idx_face[0])

    #save images

    for idx in best_poses_idx :
        cv2.imwrite("../pose"+str(idx)+".jpg", cv2.cvtColor( images[idx], cv2.COLOR_RGB2BGR))
    print(f'{best_poses_idx}poses index')
    return best_poses_idx #best_poses_idx#index_best_poses # list of the index of the best poses


# if __name__ == '__main__':

#     gif = video_to_gif('../../raw_data/input_video/IMG_1525.MOV')
#     keypoints_sequence, output_images= run_model_gif(gif)
#     best_poses_idx=  get_best_poses(keypoints_sequence,output_images)
