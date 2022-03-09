from tensorflow.keras.utils import CustomObjectScope
import tensorflow as tf
import cv2
import numpy as np
import os
# from paittern import paittern_package
# from paittern.paittern_package import *
""" from paittern.paittern_package.keypoint_model import run_model_gif
from paittern.paittern_package.video_gif import video_to_gif
from selection_pose import get_best_poses """

with CustomObjectScope():
    model = tf.keras.models.load_model("./paittern/contouring/final_model")


def predict_mask(image,model):

    l_image = image.shape[1]
    h_image = image.shape[0]
    #print(l_image, h_image)

    #Transformation de la taille de l'image
    H = 288
    W = 288
    new_image = cv2.resize(image, (H,W))/255
    new_image = np.flip(new_image, axis = -1)

    #Ajouter une dimension
    new_image = np.expand_dims(new_image, axis = 0)
    #print(new_image.shape)

    #Faire la prediction
    mask_pred = model.predict(new_image)

    #Convertir les valeurs en 0 ou 1
    mask_pred = np.where(mask_pred>0.5, 1, 0)
    mask_pred = mask_pred.squeeze()
    #print(mask_pred.shape)
    #print(np.unique(mask_pred))

    #Resizer
    mask_pred_resized = cv2.resize(mask_pred, (l_image, h_image), interpolation = cv2.INTER_NEAREST)
    #print(mask_pred_resized.shape)
    return mask_pred_resized

if __name__ == '__main__':
    with CustomObjectScope():
        model = tf.keras.models.load_model("./paittern/contouring/final_model")

    # gif = video_to_gif('../../raw_data/input_video/IMG_1525.MOV')
    # keypoints_sequence, output_images= run_model_gif(gif)
    # best_poses_idx=  get_best_poses(keypoints_sequence,output_images)
    # print(os.cwd())
    image = cv2.imread("to_test2.jpg")
    mask = predict_mask(image, model)
    mask_readable = mask*255
    print(mask_readable)
    print(np.unique(mask_readable))
    cv2.imwrite("mask.png", mask_readable)
