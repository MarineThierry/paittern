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

    return best_poses_idx #best_poses_idx#index_best_poses # list of the index of the best poses


# if __name__ == '__main__':

#     gif = video_to_gif('../../raw_data/input_video/IMG_1525.MOV')
#     keypoints_sequence, output_images= run_model_gif(gif)
#     best_poses_idx=  get_best_poses(keypoints_sequence,output_images)





# #reshape keypoints_sequence
# def resize_matrix_keypoints(matrix_keypoints):
#     return np.reshape(np.array(matrix_keypoints),(np.array(matrix_keypoints).shape[0],17,3))



# #PICTURE FRONT
# # 1st criteria calculate abs distance (on x) between 'left_shoulder', 'right_shoulder'
# def distance_x_shoulders(image_serie, matrix_keypoints):
#     distance_shoulder = []
#     for image in matrix_keypoints:
#         distance_shoulder.append(abs(image[5][1]-image[6][1]))

#     # Sorting from Max to Min distance between shoulders on x
#     list_shoulderx_sort = sorted(distance_shoulder, reverse=True)

#     #Image index based on ranking for max(x) for shoulders
#     image_index_ranking_x = []
#     for i in list_shoulderx_sort:
#         image_index_ranking_x.append(distance_shoulder.index(i))

#     #List of n images index based on criteria 1(max(x))
#     image_index_ranking_x_selection = image_index_ranking_x[ 0 : nb_image_criteria_max_x]

#     #List of keypoints (after selection with criteria 1) to apply 2nd criteria
#     keypoints_sequence_1_crit_1 = []
#     for i in image_index_ranking_x_selection:
#         keypoints_sequence_1_crit_1.append(keypoints_sequence_1[i])

#     return  image_serie[image_index_ranking_x_selection[0]], matrix_keypoints[image_index_ranking_x_selection[0]], keypoints_sequence_1_crit_1





# #----------------------------------------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------------------------------------

# #PICTURE FRONT (after applyring criteria 1)
# #2nd criteria (2b) to minimize distance between y : standard deviation between 6 keypoints(shoulders,elbows,wrists)
# def distance_min_y_shou_elb_wrist(image_serie, matrix_keypoints):
#     horizontal_selection_2b = []
#     for image in keypoints_sequence_1_crit_1:
#         a  = np.std([image[5][0],image[6][0], image[7][0], image[8][0], image[9][0], image[10][0]])
#         horizontal_selection_2b.append(a)

#     #Min standard deviation - keypoints sequence in keypoints_sequence_1_crit_1
#     min_sdt = keypoints_sequence_1_crit_1[horizontal_selection_2b.index(min(horizontal_selection_2b))]

#     #Image index in keypoints_sequence_1
#     for idx, arr in enumerate(keypoints_sequence_1):
#         comparaison = arr == min_sdt
#         if comparaison.all():
#             image_index_selection_2b = idx
#             break

#     return image_serie[image_index_selection_2b], matrix_keypoints[image_index_selection_2b]



# #----------------------------------------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------------------------------------

# #PICTURE PROFILE
# #Criteria 3: minimize distance between x : standard deviation between 8 keypoints (shoulders/hips/knees/ankles)

# def distance_min_x_shoul_hip_knee_ankle(image_serie, matrix_keypoints):
#     profil_selection = []
#     for image in matrix_keypoints:
#         a  = np.std([image[5][1],image[6][1], image[11][1], image[12][1], image[13][1], image[14][1], image[15][1], image[16][1]] )
#         profil_selection.append(a)
#     profil_selection_sorted = sorted(profil_selection, reverse=False)

#     #List of images index based on ranking for min(standard deviation) on x: shoulders/hips/knees/ankles
#     image_index_profil_a = []
#     for i in profil_selection_sorted:
#         image_index_profil_a.append(profil_selection.index(i))

#     return image_serie[image_index_profil_a[0]], matrix_keypoints[image_index_profil_a[0]]




# if __name__ == '__main__':

#     gif = video_to_gif('../raw_data/input_video/test_keypoint_v1.mov')
#     keypoints_sequence, output_images= run_model_gif(gif)

#     keypoints_sequence_1 = resize_matrix_keypoints(keypoints_sequence)
#     #Definition of list of keypoints (after selection with criteria 1) to apply 2nd criteria
#     keypoints_sequence_1_crit_1 = distance_x_shoulders(output_images, keypoints_sequence_1)[2]

#     #display best Picture Front with criteria 1 + keypoints matrix
#     plt.imshow(distance_x_shoulders(output_images, keypoints_sequence_1)[0])
#     distance_x_shoulders(output_images, keypoints_sequence_1)[1]

#     #display best Picture Front with criteria 1 + keypoints matrix
#     plt.imshow(distance_x_shoulders(output_images, keypoints_sequence_1)[0])
#     distance_x_shoulders(output_images, keypoints_sequence_1)[1]

#     #display best picture front with criteria 1 + critera 2 + keypoints matrix
#     plt.imshow(distance_min_y_shou_elb_wrist(output_images, keypoints_sequence_1)[0]),
#     distance_min_y_shou_elb_wrist(output_images, keypoints_sequence_1)[1]

#     #display best profile image with criteria 3 + keypoints matrix
#     plt.imshow(distance_min_x_shoul_hip_knee_ankle(output_images, keypoints_sequence_1)[0])
#     distance_min_x_shoul_hip_knee_ankle(output_images, keypoints_sequence_1)[1]

#     #display best profile image with criteria 3 + keypoints matrix
#     plt.imshow(distance_min_x_shoul_hip_knee_ankle(output_images, keypoints_sequence_1)[0])
#     distance_min_x_shoul_hip_knee_ankle(output_images, keypoints_sequence_1)[1]
