from re import M
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

def additional_kp_front(image_front_T_kp, image_front_I_kp):
    #Biceps - Image front T
    #left biceps pos 17
    l_x11 = image_front_T_kp[5][1] +abs(image_front_T_kp[7][1] - image_front_T_kp[5][1])/2
    l_y11 = (image_front_T_kp[7][0] + image_front_T_kp[5][0])/2
    #right biceps pos 18
    r_x11 = image_front_T_kp[8][1] + abs(image_front_T_kp[8][1]-image_front_T_kp[6][1])/2
    r_y11 = (image_front_T_kp[8][0] + image_front_T_kp[6][0])/2

    #Biceps - Image front I
    #left biceps pos 17
    l_x12 = abs(image_front_I_kp[7][1] + image_front_I_kp[5][1])/2
    l_y12 = image_front_I_kp[5][0] + abs(image_front_I_kp[7][0] - image_front_I_kp[5][0])/2
    #right biceps pos 18
    r_x12 = abs(image_front_I_kp[8][1]+image_front_I_kp[6][1])/2
    r_y12 = image_front_I_kp[6][0] + (image_front_I_kp[8][0] + image_front_I_kp[6][0])/2


    #left chest, right chest                      pos 19 / 20
    #left chest pos 19
    l_x2 = image_front_T_kp[6][1] + abs(image_front_T_kp[6][1] - image_front_T_kp[5][1]) * 1/4   #considering chest "x" located  at 1/4 vs soulders
    l_y2 = image_front_T_kp[5][0] + abs(image_front_T_kp[11][0] - image_front_T_kp[5][0]) * 1/3  #considering chest "y" located  at 1/3 vs shoulders/hips
    #right chest pos 20
    r_x2 = image_front_T_kp[6][1] + abs(image_front_T_kp[6][1] - image_front_T_kp[5][1]) * 3/4   #considering chest "x" located  at 3/4 vs soulders
    r_y2 = image_front_T_kp[6][0] + abs(image_front_T_kp[12][0] - image_front_T_kp[6][0]) * 1/3  #considering chest "y" located  at 1/3 vs shoulders/hips

    #left waist, right waist                       pos 21 / 22
    #left waist pos 21
    l_x3 = l_x2                                                                                   #considering chest "x" located  at 1/4 vs soulders
    l_y3 = image_front_T_kp[5][0] + abs(image_front_T_kp[11][0] - image_front_T_kp[5][0]) * 2/3  #considering chest "y" located  at 2/3 vs shoulders/hi
    #right waist pos 22
    r_x3 = r_x2                                                                                   #considering chest "x" located  at 3/4 vs soulders
    r_y3 = image_front_T_kp[6][0] + abs(image_front_T_kp[12][0] - image_front_T_kp[6][0]) * 2/3  #considering chest "y" located  at 2/3 vs shoulders/hips

    #append new keypoints into matrix_kp front T, front I
    new_matrix_kp_T = np.vstack((image_front_T_kp, np.array([l_y11, l_x11, 0]), np.array([r_y11, r_x11, 0]), \
                              np.array([l_y2, l_x2, 0]), np.array([r_y2, r_x2, 0]), \
                              np.array([l_y3, l_x3, 0]), np.array([r_y3, r_x3, 0])))

    new_matrix_kp_I = np.vstack((image_front_I_kp, np.array([l_y12, l_x12, 0]), np.array([r_y12, r_x12, 0]), \
                              np.array([l_y2, l_x2, 0]), np.array([r_y2, r_x2, 0]), \
                              np.array([l_y3, l_x3, 0]), np.array([r_y3, r_x3, 0])))


    return new_matrix_kp_T, new_matrix_kp_I

def denormalize(matrix, image_h, image_l):
    x = np.round(matrix[: , 1] * image_l)
    x = [ int(i) for i in x]                                      #convert in int
    y = np.round(matrix[: , 0] * image_h)
    y = [ int(i) for i in y]                                      #convert in int
    matrix_kp_converted = np.array([x,y])
    return matrix_kp_converted.T

def ratio_real_vs_pixel(real_height, matrix_image_contouring):
    list_sum_y=[]
    #sum of all "1" for each x to get the sum max ie. height
    for i in range(0,matrix_image_contouring.shape[1]):
        list_sum_y.append(np.sum(matrix_image_contouring[: , i]))
    ratio_mm_px = real_height / max(list_sum_y)                            #ratio real height (in mm) / max sum of "1"
    return ratio_mm_px

def sum_segment(origin, direction, image):
    x_origin, y_origin = origin
    if direction == 'height':
        y = y_origin
        while image[y][x_origin]>0:
            y_top = y
            y += 1
        y = y_origin
        while image[y][x_origin]>0:
            y_bottom = y
            y -= 1
        return y_top - y_bottom + 1
    if direction == 'width':
        x = x_origin
        while image[y_origin][x]>0:
            x_right = x
            x += 1
        x = x_origin
        while image[y_origin][x]>0:
            x_left = x
            x -= 1
        return x_right - x_left + 1

#BICEPS CIRCUMFERENCE _ Front picture in T shape
def biceps_circumference(ratio_mm_px, matrix_kp_front_converted, matrix_image_front_cont, matrix_kp_profile_converted=None, matrix_image_profile_cont=None):
    #left biceps
    sum_left = sum_segment(matrix_kp_front_converted[17], 'height', matrix_image_front_cont)

    #right biceps
    sum_right = sum_segment(matrix_kp_front_converted[18], 'height', matrix_image_front_cont)

    #diameter left / right biceps "height"
    average_mm = ((sum_left + sum_right) / 2) * ratio_mm_px

    #biceps circumference in mm
    circumference_biceps = (average_mm) * math.pi
    return  round(circumference_biceps)

#CHEST CIRCUMFERENCE _ Front picture in T shape + profile picture
def chest_circumference(ratio_mm_px, matrix_kp_front_converted, matrix_image_front_cont,
                        matrix_kp_profile_converted=None, matrix_image_profile_cont=None):
    #measure width (along axis x) for given y, on front picture
    y_front1 = sum_segment(matrix_kp_front_converted[19], 'width', matrix_image_front_cont)
    y_front2 = sum_segment(matrix_kp_front_converted[20], 'width', matrix_image_front_cont)
    sum_x_front = int((y_front1+y_front2)/2)

    #measure width (along axis x) for given y, on profile picture
    #y_prof1 = sum_segment(matrix_kp_front_converted[19], 'width', matrix_image_profile_cont)
    #y_prof2 = sum_segment(matrix_kp_front_converted[20], 'width', matrix_image_profile_cont)
    #sum_x_prof = int((y_prof1 + y_prof2)/2)
    sum_x_prof = sum_x_front / 2

    #chest circumference in mm
    circumference_chest = (sum_x_front + sum_x_prof) * 2 * ratio_mm_px

    return round(circumference_chest)

#HPS TO WAIST BACK _ Current solution is distance between shoulder (# 5 / 6) and waist (# 21 / 22) with a factor 1.2
def hps_to_waist_back(ratio_mm_px, matrix_kp_front_converted, matrix_image_front_cont, matrix_kp_profile_converted=None, matrix_image_profile_cont=None):
    a = (abs(matrix_kp_front_converted[5][0] - matrix_kp_front_converted[21][0]) + abs(matrix_kp_front_converted[6][0] - matrix_kp_front_converted[22][0])) /2
    #convert in mm and apply factor of 1.2
    distance = a * 1.2 * ratio_mm_px
    return round(distance)


#HIPS_CIRCUMFERENCE _ Front picture in T shape + Profile picture
def hips_circumference(ratio_mm_px, matrix_kp_front_converted, matrix_image_front_cont, matrix_kp_profile_converted=None, matrix_image_profile_cont=None): #matrix_kp_profile_converted, matrix_image_profile_cont):

    #measure width (along axis x) for given y, on front picture
    y_front1 = sum_segment(matrix_kp_front_converted[11], 'width', matrix_image_front_cont)
    y_front2 = sum_segment(matrix_kp_front_converted[12], 'width', matrix_image_front_cont)
    sum_x_front = int((y_front1+y_front2)/2)

    #measure width (along axis x) for given y, on profile picture
    #y_prof1 = sum_segment(matrix_kp_front_converted[11], 'width', matrix_image_profile_cont)
    #y_prof2 = sum_segment(matrix_kp_front_converted[12], 'width', matrix_image_profile_cont)
    #sum_x_prof = int((y_prof1 + y_prof2)/2)
    sum_x_prof = sum_x_front

    #hips circumference in mm
    hips_circ = (sum_x_front + sum_x_prof) * 2 * ratio_mm_px
    return round(hips_circ)

#NECK_CIRCUMFERENCE _ Profile picture
def neck_circumference(ratio_mm_px, matrix_kp_front_converted, matrix_image_front_cont, matrix_kp_profile_converted=None, matrix_image_profile_cont=None):
    #Estimated neck position on y located at 1/3 above shoulders vs ears
    y_shoulder = matrix_kp_front_converted[6][1]
    y_head = matrix_kp_front_converted[0][1]
    width_list = []
    for y in range(y_head,y_shoulder):
        width = sum_segment((matrix_kp_front_converted[0][0],y), 'width', matrix_image_front_cont)
        width_list.append(width)

    neck_width = sorted(width_list)[0]

    #neck circumference in mm
    neck_circ = (neck_width * ratio_mm_px) * math.pi
    return round(neck_circ)

#SHOULDER_SLOPE _ Front picture in T shape
#Consider distance (on x) between shoulder kp and ear kp, plus half width of biceps
def shoulder_slope(ratio_mm_px, matrix_kp_front_converted, matrix_image_front_cont, matrix_kp_profile_converted=None, matrix_image_profile_cont=None):
    return 0.314

#SHOULDER TO SHOULDER _ Front picture in I shape
def shoulder_to_shoulder(ratio_mm_px, matrix_kp_front_converted, matrix_image_front_cont, matrix_kp_profile_converted=None, matrix_image_profile_cont=None):
    #average y for shoulders kp
    y1 =int(( matrix_kp_front_converted[5][1] + matrix_kp_front_converted[6][1] ) / 2)
    #distance (on x) between shoulders
    sum_shoulders = np.sum(matrix_image_front_cont[y1, :])
    #convert distance in mm
    distance = sum_shoulders * ratio_mm_px
    return round(distance)

# WAIST CIRCUMFERENCE _ Front picture in T shape + Profile picture
def waist_circumference(ratio_mm_px, matrix_kp_front_converted, matrix_image_front_cont, matrix_kp_profile_converted=None, matrix_image_profile_cont=None):

    #measure width (along axis x) for given y, on front picture
    y_front1 = sum_segment(matrix_kp_front_converted[21], 'width', matrix_image_front_cont)
    y_front2 = sum_segment(matrix_kp_front_converted[22], 'width', matrix_image_front_cont)
    sum_x_front = int((y_front1+y_front2)/2)

    #measure width (along axis x) for given y, on profile picture
    #y_prof1 = sum_segment(matrix_kp_front_converted[21], 'width', matrix_image_profile_cont)
    #y_prof2 = sum_segment(matrix_kp_front_converted[22], 'width', matrix_image_profile_cont)
    #sum_x_prof = int((y_prof1 + y_prof2)/2)
    sum_x_prof = sum_x_front / 2

    #waist circumference in mm
    waist_circumference = (sum_x_front + sum_x_prof) * 2 * ratio_mm_px
    return round(waist_circumference)

#WAIST_TO_HIPS _ Front picture in T shape
def waist_to_hips(ratio_mm_px, matrix_kp_front_converted, matrix_image_front_cont, matrix_kp_profile_converted=None, matrix_image_profile_cont=None):
    #average distance (on y) between waist(21,22) and hips(11,12) with correction factor of 1.2
    a = (abs(matrix_kp_front_converted[22][1] - matrix_kp_front_converted[12][1])
         +abs(matrix_kp_front_converted[21][1] - matrix_kp_front_converted[11][1])) / 2
    #convert in mm and apply factor of 1.2
    distance = a * 1.2 * ratio_mm_px
    return round(distance)

#Dictionnary of required measures for each pattern
dico_pattern_measures={
    'aaron': {'biceps':biceps_circumference,
              'chest':chest_circumference,
              'hpsToWaistBack':hps_to_waist_back,
              'hips':hips_circumference,
              'neck':neck_circumference,
              'shoulderSlope':shoulder_slope,
              'shoulderToShoulder':shoulder_to_shoulder,
              'waistToHips':waist_to_hips},
}

def get_measures(pattern,
                 ratio_mm_px,
                 matrix_kp_front_converted,
                 matrix_image_front_cont,
                 matrix_kp_profile_converted=None,
                 matrix_image_profile_cont=None):
    measures = {}
    for name, function in dico_pattern_measures[pattern].items():
        measures[name]=function(ratio_mm_px, matrix_kp_front_converted, matrix_image_front_cont)

    return measures

def from_pix2measures(pattern,
                      real_height,
                     front_T_kp,
                     #front_I_kp,
                     #profile_kp,
                     front_T_cont,
                     #front_I_cont,
                     #profile_cont
                     ):
    print('Enter measures model')
    # On ajoute des kp additionnels #
    front_T_kp_plus, front_I_kp_plus = additional_kp_front(front_T_kp, front_T_kp)

    # On dénormalise les matrices de kp #
    front_T_kp_plus = denormalize(front_T_kp_plus, front_T_cont.shape[0], front_T_cont.shape[1])
    #front_I_kp_plus = denormalize(front_I_kp_plus, front_I_cont.shape[0], front_I_cont.shape[1])
    #profile_kp_plus = denormalize(profile_kp_plus, profile_cont.shape[0], profile_cont.shape[1])

    # On calcule la valeur d'un pixel en mm #
    ratio_mm_px = ratio_real_vs_pixel(real_height, front_T_cont)

    # On obtient les mesures pour un pattern spécifique #
    dico_measures = get_measures(pattern,ratio_mm_px, front_T_kp_plus, front_T_cont)

    return dico_measures
