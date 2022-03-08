from re import M
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2



#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------


# array = cv2.imread(r"pexels-photo-60219.png")
# matrix_front_T_cont = array.mean(axis=-1)
# print(matrix_front_T_cont.shape)

# test_matrix_front_kp = np.array([[0.48695812, 0.68326586, 0.5639537 ],
#           [0.4778103 , 0.68713045, 0.50732034],
#           [0.47737232, 0.67476934, 0.5956619 ],
#           [0.4883077 , 0.70074975, 0.57466257],
#           [0.4893417 , 0.6668179 , 0.49692088],
#           [0.52557784, 0.7172219 , 0.48050335],
#           [0.539968  , 0.6626774 , 0.5938845 ],
#           [0.52591187, 0.7609507 , 0.21174023],
#           [0.54629296, 0.63570327, 0.28719622],
#           [0.51536655, 0.852199  , 0.3918358 ],
#           [0.51517344, 0.62228054, 0.13233644],
#           [0.6867487 , 0.70954204, 0.4412909 ],
#           [0.68755573, 0.6688813 , 0.366415  ],
#           [0.8038848 , 0.72793126, 0.46259767],
#           [0.7948738 , 0.6616839 , 0.4294906 ],
#           [0.9306485 , 0.737101  , 0.3557214 ],
#           [0.9053455 , 0.621957  , 0.37097037]])

# #New matrix with new KP
# additional_kp_front(test_matrix_front_kp,test_matrix_front_kp )
# test_matrix_front_kp = additional_kp_front(test_matrix_front_kp,test_matrix_front_kp )[0]
# test_matrix_front_kp_1 =  additional_kp_front(test_matrix_front_kp,test_matrix_front_kp )[1]
# #New matrix with new KP + converted into pixels coordinates
# test_matrix_front_kp_conv = convert_kp_matrix_front_T(test_matrix_front_kp, 768, 1024)
# test_matrix_front_kp_1_conv = convert_kp_matrix_front_I(test_matrix_front_kp, 768, 1024)


#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

image_l = matrix_front_T_cont.shape[1]
image_h = matrix_front_T_cont.shape[0]


#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#ADDITIONAL KEYPOINTs

#ADDITIONAL KEYPOINTS on picture FRONT_T / FRONT_I
#left biceps, right biceps                      pos 17 / 18

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


#ADDITIONAL KEYPOINTS on picture PROFILE
#left biceps, right biceps                      pos 17 / 18

def additional_kp_profile(image_profile_kp):
    #left biceps pos 17
    l_x1 = (image_profile_kp[7][1] + image_profile_kp[5][1])/2                            #average on x
    l_y1 = image_profile_kp[5][0] + (image_profile_kp[7][0] - image_profile_kp[5][0])/2   #considering biceps is at half shoulder/elbow
    #right biceps pos 18
    r_x1 = abs(image_profile_kp[8][1] + image_profile_kp[6][1])/2                         #average on x
    r_y1 = image_profile_kp[6][0] + (image_profile_kp[8][0] - image_profile_kp[6][0])/2   #considering biceps is at half shoulder/elbow

    #left chest, right chest                      pos 19 / 20
    #left chest pos 19
    l_x2 = l_x1
    l_y2 = image_profile_kp[5][0] + abs(image_profile_kp[11][0] - image_profile_kp[5][0]) * 1/3  #considering chest "y" located  at 1/3 vs shoulders/hips
    #right chest pos 20
    r_x2 = r_x1
    r_y2 = image_profile_kp[6][0] + abs(image_profile_kp[12][0] - image_profile_kp[6][0]) * 1/3  #considering chest "y" located  at 1/3 vs shoulders/hips

    #left waist, right waist                       pos 21 / 22
    #left waist pos 21
    l_x3 = l_x1
    l_y3 = image_profile_kp[5][0] + abs(image_profile_kp[11][0] - image_profile_kp[5][0]) * 2/3  #considering chest "y" located  at 2/3 vs shoulders/hi
    #right waist pos 22
    r_x3 = r_x1
    r_y3 = image_profile_kp[6][0] + abs(image_profile_kp[12][0] - image_profile_kp[6][0]) * 2/3  #considering chest "y" located  at 2/3 vs shoulders/hips

    #append new keypoints into matrix_kp
    new_matrix_kp_profile = np.vstack((image_profile_kp, np.array([l_y1, l_x1, 0]), np.array([r_y1, r_x1, 0]), \
                              np.array([l_y2, l_x2, 0]), np.array([r_y2, r_x2, 0]), \
                              np.array([l_y3, l_x3, 0]), np.array([r_y3, r_x3, 0])))


    return new_matrix_kp_profile


#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

#CONVERT MATRIX KP coordinates in x,y to pixels(contouring matrix)
#Image front T
def convert_kp_matrix_front_T(matrix_kp_front_T, image_h, image_l):
    x = np.round(matrix_kp_front_T[: , 1] * image_l)
    x = [ int(i) for i in x]                                      #convert in int
    y = np.round(matrix_kp_front_T[: , 0] * image_h)
    y = [ int(i) for i in y]                                      #convert in int
    matrix_kp_converted = list(zip(y,x))
    return matrix_kp_converted
#Image front I
def convert_kp_matrix_front_I(matrix_kp_front_I, image_h, image_l):
    x = np.round(matrix_kp_front_I[: , 1] * image_l)
    x = [ int(i) for i in x]
    y = np.round(matrix_kp_front_I[: , 0] * image_h)
    y = [ int(i) for i in y]
    matrix_kp_converted = list(zip(y,x))
    return matrix_kp_converted
#Image profile
def convert_kp_matrix_profile(matrix_kp_profile, image_h, image_l):
    x = np.round(matrix_kp_profile[: , 1] * image_l)
    x = [ int(i) for i in x]
    y = np.round(matrix_kp_profile[: , 0] * image_h)
    y = [ int(i) for i in y]
    matrix_kp_converted = list(zip(y,x))
    return matrix_kp_converted

# Define ratio between real measure(in mm) vs pixel position
def ratio_real_vs_pixel(real_height, matrix_image_contouring, image_l):
    list_sum_y=[]
    #sum of all "1" for each x to get the sum max ie. height
    for i in range(0,image_l):
        list_sum_y.append(np.sum(matrix_image_contouring[: , i]))
    print(max(list_sum_y))
    ratio_mm_px = real_height / max(list_sum_y)                            #ratio real height (in mm) / max sum of "1"
    return ratio_mm_px


ratio_mm_px = ratio_real_vs_pixel(real_height, matrix_profile_cont, image_l)

#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

#BICEPS CIRCUMFERENCE _ Front picture in T shape
def biceps_circumference(matrix_kp_front_converted, matrix_image_front_cont):
    #left biceps
    x1 = matrix_kp_front_converted[17][1]
    sum_left = np.sum(matrix_image_front_cont[:, x1])

    #right biceps
    x2 = matrix_kp_front_converted[18][1]
    sum_right = np.sum(matrix_image_front_cont[:, x2])

    #diameter left / right biceps "height"
    average_mm = ((sum_left + sum_right) / 2) * ratio_mm_px

    #biceps circumference in mm
    circumference_biceps = (average_mm) * math.pi
    print(x1,x2)
    print(sum_left, sum_right)
    print(average_mm)
    return  circumference_biceps

#CHEST CIRCUMFERENCE _ Front picture in T shape + profile picture
def chest_circumference(matrix_kp_front_converted, matrix_image_front_cont, matrix_kp_profile_converted, matrix_image_profile_cont, ):
    #measure width (along axis x) for given y, on front picture
    y_front1 = matrix_kp_front_converted[19][0]
    y_front2 = matrix_kp_front_converted[20][0]
    y_average = int((y_front1+y_front2)/2)
    sum_x_front = np.sum(matrix_image_front_cont[y_average,:])

    #measure width (along axis x) for given y, on profile picture
    y_prof1 = matrix_kp_front_converted[19][0]
    y_prof2 = matrix_kp_front_converted[20][0]
    y_average = int((y_prof1 + y_prof2)/2)
    sum_x_prof = np.sum(matrix_image_front_cont[y_average,:])

    #chest circumference in mm
    circumference_chest = (sum_x_front + sum_x_prof) * 2 * ratio_mm_px

    print(y_front1, y_front2)
    print(matrix_image_front_cont[y_average,:])
    print(sum_x_front, sum_x_prof)
    return circumference_chest

#HPS TO WAIST BACK _ Current solution is distance between shoulder (# 5 / 6) and waist (# 21 / 22) with a factor 1.2
def hps_to_waist_back(matrix_kp_front_converted, matrix_image_front_cont):
    a = (abs(matrix_kp_front_converted[5][0] - matrix_kp_front_converted[21][0]) + abs(matrix_kp_front_converted[6][0] - matrix_kp_front_converted[22][0])) /2
    #convert in mm and apply factor of 1.2
    distance = a * 1.2 * ratio_mm_px
    return distance


#HIPS_CIRCUMFERENCE _ Front picture in T shape + Profile picture
def hips_circumference(matrix_kp_front_converted, matrix_image_front_cont, matrix_kp_profile_converted, matrix_image_profile_cont):

    #measure width (along axis x) for given y, on front picture
    y_front1 = matrix_kp_front_converted[11][1]
    y_front2 = matrix_kp_front_converted[12][1]
    y_average = int((y_front1+y_front2)/2)
    sum_x_front = np.sum(matrix_image_front_cont[y_average,:])

    #measure width (along axis x) for given y, on profile picture
    y_prof1 = matrix_kp_front_converted[11][1]
    y_prof2 = matrix_kp_front_converted[12][1]
    y_average = int((y_prof1 + y_prof2)/2)
    sum_x_prof = np.sum(matrix_image_front_cont[y_average,:])

    #hips circumference in mm
    hips_circ = (sum_x_front + sum_x_prof) * 2 * ratio_mm_px
    return hips_circ


#NECK_CIRCUMFERENCE _ Profile picture
def neck_circumference( matrix_kp_profile_converted, matrix_image_profile_cont):
    #Estimated neck position on y located at 1/3 above shoulders vs ears
    neck_y = int(matrix_kp_profile_converted[5][0]
                 - abs(matrix_kp_profile_converted[5][0] - matrix_kp_profile_converted[3][0]) *(1/4))
    #sum over x considering neck_y = a
    sum_x = np.sum(matrix_image_profile_cont[neck_y,:])

    #neck circumference in mm
    neck_circ = (sum_x * ratio_mm_px) * math.pi
    return neck_circ

#SHOULDER_SLOPE _ Front picture in T shape
#Consider distance (on x) between shoulder kp and ear kp, plus half width of biceps
def shoulder_slope(matrix_kp_front_converted, matrix_image_front_cont):
    #left biceps
    x1 = matrix_kp_front_converted[17][1]
    sum_left = np.sum(matrix_image_front_cont[:, x1])

    #distance (on x) between kp for shoulder / ear
    distance = (abs(matrix_kp_front_converted[5][1] - matrix_kp_front_converted[3][1]) + abs(matrix_kp_front_converted[4][1] - matrix_kp_front_converted[6][1])) / 2

    #overall distance in mm
    total_distance = distance + sum_left/2 * ratio_mm_px
    return total_distance

#SHOULDER TO SHOULDER _ Front picture in I shape
def shoulder_to_shoulder( matrix_kp_front_converted, matrix_image_front_cont ):
    #average y for shoulders kp
    y1 =int(( matrix_kp_front_converted[5][1] + matrix_kp_front_converted[6][1] ) / 2)
    #distance (on x) between shoulders
    sum_shoulders = np.sum(matrix_image_front_cont[y1, :])
    #convert distance in mm
    distance = sum_shoulders * ratio_mm_px
    return distance


# WAIST CIRCUMFERENCE _ Front picture in T shape + Profile picture
def waist_circumference(matrix_kp_front_converted, matrix_image_front_cont, matrix_kp_profile_converted, matrix_image_profile_cont):

    #measure width (along axis x) for given y, on front picture
    y_front1 = matrix_kp_front_converted[21][0]
    y_front2 = matrix_kp_front_converted[22][0]
    y_average = int((y_front1+y_front2)/2)
    sum_x_front = np.sum(matrix_image_front_cont[y_average,:])

    #measure width (along axis x) for given y, on profile picture
    y_prof1 = matrix_kp_profile_converted[11][0]
    y_prof2 = matrix_kp_profile_converted[12][0]
    y_average = int((y_prof1 + y_prof2)/2)
    sum_x_prof = np.sum(matrix_image_front_cont[y_average,:])

    #waist circumference in mm
    waist_circumference = (sum_x_front + sum_x_prof) * 2 * ratio_mm_px
    return waist_circumference

#WAIST_TO_HIPS _ Front picture in T shape
def waist_to_hips(matrix_kp_front_converted, matrix_image_front_cont):
    #average distance (on y) between waist(21,22) and hips(11,12) with correction factor of 1.2
    a = (abs(matrix_kp_front_converted[22][1] - matrix_kp_front_converted[12][1])
         +abs(matrix_kp_front_converted[21][1] - matrix_kp_front_converted[11][1])) / 2
    #convert in mm and apply factor of 1.2
    distance = a * 1.2 * ratio_mm_px
    return distance


#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

#Dictionnary of required measures for each pattern
dico_pattern_measures={
    'Aaron': ['Biceps circumference','Chest circumference','HPS to waist back','Hips circumference','Neck circumference',
              'Shoulder slope','Shoulder to shoulder','Waist to hips'],
}

#Dictionnary of measure / associated function
dico_measures = {
    'Biceps circumference' : [biceps_circumference(matrix_front_T_kp_conv, matrix_front_T_cont)],
    'Chest circumference' : [chest_circumference(matrix_front_T_kp_conv, matrix_front_T_cont,
                                                 matrix_profile_kp_conv, matrix_profile_cont, )],
    'HPS to waist back' : [hps_to_waist_back(matrix_front_T_kp_conv, matrix_front_T_cont)],
    'Hips circumference' : [hips_circumference(matrix_front_T_kp_conv, matrix_front_T_cont,
                                                 matrix_profile_kp_conv, matrix_profile_cont, )],
    'Neck circumference' : [neck_circumference(matrix_profile_kp_conv, matrix_profile_cont,)],
    'Shoulder slope' : [shoulder_slope(matrix_front_T_kp_conv, matrix_front_T_cont)],
    'Shoulder to shoulder': [shoulder_to_shoulder(matrix_front_I_kp_conv, matrix_front_I_cont)],
    'Waist to hips' : [waist_to_hips(matrix_front_T_kp_conv, matrix_front_T_cont)]
}

def get_measures(pattern):
    measures = {}
    for i in range(0,len(dico_pattern_measures[pattern])):
        #print(measures[i])
        measures[dico_pattern_measures[pattern][i]] = dico_measures[dico_pattern_measures[pattern][i]]

    return measures

#Test
get_measures('Aaron')
