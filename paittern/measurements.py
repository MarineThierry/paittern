import numpy as np
import math

#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#ADDITIONAL KEYPOINTS

#left biceps, right biceps                  pos 17 / 18

def biceps_kp(image_front_T_kp, matrix_kp):
    #left biceps pos 17
    l_x = abs(image_front_T_kp[7][1] - image_front_T_kp[5][1])
    l_y = (image_front_T_kp[7][0] + image_front_T_kp[5][0])/2
    #right biceps pos 18
    r_x = abs(image_front_T_kp[8][1]-image_front_T_kp[6][1])
    r_y = (image_front_T_kp[8][0] + image_front_T_kp[6][0])/2

    #append new keypoints into matrix_kp
    new_matrix_kp = np.append(matrix_kp, [l_y, l_x, 0], [r_y, r_x, 0], axis=0)
    return new_matrix_kp

#left chest, right chest                      pos 19 / 20
def chest_kp(image_front_T_kp, matrix_kp):
    #left chest pos 19
    l_x = image_front_T_kp[5][1] + abs(image_front_T_kp[6][1] - image_front_T_kp[5][1]) * 1/4   #considering chest "x" located  at 1/4 vs soulders
    l_y = image_front_T_kp[5][0] + abs(image_front_T_kp[11][0] - image_front_T_kp[5][0]) * 1/3  #considering chest "y" located  at 1/3 vs shoulders/hips
    #right chest pos 20
    r_x = image_front_T_kp[5][1] + abs(image_front_T_kp[6][1] - image_front_T_kp[5][1]) * 3/4   #considering chest "x" located  at 3/4 vs soulders
    r_y = image_front_T_kp[6][0] + abs(image_front_T_kp[12][0] - image_front_T_kp[6][0]) * 1/3  #considering chest "y" located  at 1/3 vs shoulders/hips
    #append new keypoints into matrix_kp
    new_matrix_kp = np.append(matrix_kp, [l_y, l_x, 0], [r_y, r_x, 0], axis=0)
    return new_matrix_kp

#left waist, right waist                       pos 21 / 22
def waist_kp(image_front_T_kp, matrix_kp):
    #left waist pos 21
    l_x = image_front_T_kp[5][1] + abs(image_front_T_kp[6][1] - image_front_T_kp[5][1]) * 1/4   #considering chest "x" located  at 1/4 vs soulders
    l_y = image_front_T_kp[5][0] + abs(image_front_T_kp[11][0] - image_front_T_kp[5][0]) * 2/3  #considering chest "y" located  at 2/3 vs shoulders/hi
    #right waist pos 22
    r_x = image_front_T_kp[5][1] + abs(image_front_T_kp[6][1] - image_front_T_kp[5][1]) * 3/4   #considering chest "x" located  at 3/4 vs soulders
    r_y = image_front_T_kp[6][0] + abs(image_front_T_kp[12][0] - image_front_T_kp[6][0]) * 2/3  #considering chest "y" located  at 2/3 vs shoulders/hips
    #append new keypoints into matrix_kp
    new_matrix_kp = np.append(matrix_kp, [l_y, l_x, 0], [r_y, r_x, 0], axis=0)
    return new_matrix_kp


#Convert Matrix Keypoints coordinates in x,y to pixels(contouring matrix)
def convert_kp_matrix_front_T(matrix_kp_front_T, image_h, image_l):
    x = np.round(matrix_kp_front_T[: , 1] * image_l)
    y = np.round(matrix_kp_front_T[: , 0] * image_h)
    matrix_kp_converted = list(zip(y,x))
    return matrix_kp_converted

def convert_kp_matrix_front_I(matrix_kp_front_I, image_h, image_l):
    x = np.round(matrix_kp_front_I[: , 1] * image_l)
    y = np.round(matrix_kp_front_I[: , 0] * image_h)
    matrix_kp_converted = list(zip(y,x))
    return matrix_kp_converted

def convert_kp_matrix_profile(matrix_kp_profile, image_h, image_l):
    x = np.round(matrix_kp_profile[: , 1] * image_l)
    y = np.round(matrix_kp_profile[: , 0] * image_h)
    matrix_kp_converted = list(zip(y,x))
    return matrix_kp_converted

# Define ratio between real measure(in mm) vs pixel position
def ratio_real_vs_pixel(real_height, matrix_image_contouring, image_h):
    list_sum_y=[]
    #sum of all "1" for each x to get the sum max ie. height
    for i in range(0,image_h):
        list_sum_y.append(np.sum(matrix_image_contouring[: , i]))
    ratio_mm_px = real_height / max(list_sum_y)                            #ratio real height (in mm) / max sum of "1"
    return ratio_mm_px

ratio_mm_px = ratio_real_vs_pixel(real_height, matrix_image_contouring, image_h)



#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
