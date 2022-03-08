from tensorflow.keras.utils import CustomObjectScope
import tensorflow as tf
import cv2
import numpy as np

with CustomObjectScope():
    model = tf.keras.models.load_model("./paittern/contouring/contouring_model")


def predict_mask(image):
    
    l_image = image.shape[1]
    h_image = image.shape[0]
    print(l_image, h_image)
    
    #Transformation de la taille de l'image
    H = 288
    W = 288
    new_image = cv2.resize(image, (H,W))
    
    #Ajouter une dimension
    new_image = np.expand_dims(new_image, axis = 0)
    
    #Faire la prediction
    mask_pred = model.predict(new_image)
    
    #Convertir les valeurs en 0 ou 1
    mask_pred = np.where(mask_pred>0.5, 1, 0)
    mask_pred = mask_pred[0]
    print(mask_pred.shape)
    
    #Resizer
    mask_pred_resized = cv2.resize(mask_pred[0], (l_image, h_image), interpolation = cv2.INTER_NEAREST)
    
    return mask_pred_resized

if __name__ == '__main__':
    with CustomObjectScope():
        model = tf.keras.models.load_model("./paittern/contouring/contouring_model")
    
    image = cv2.imread('to_test2.jpg')
    mask = predict_mask(image)
    cv2.imwrite("mask.png", mask)
        