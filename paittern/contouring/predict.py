from tensorflow import keras
import cv2
import numpy as np

model = keras.models.load_model('/Users/humbert/code/MarineThierry/paittern/saved_models/contouring5')

def predict_mask(image):
    
    l_image = image.shape[1]
    h_image = image.shape[0]
    
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
    
    #Resizer
    mask_pred_resized = cv2.resize(mask_pred, (l_image, h_image))
    
    #Créer le collage
    
    
    #Retourner le mask et le collage
    pass