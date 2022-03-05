from tensorflow import keras

model = keras.models.load_model('../../model.h5')

def predict_mask(image):
    #Transformation d'image en 512x512
    #Ajouter une dimension
    #Faire la prediction
    #Convertir les valeurs en 0 ou 1
    #Resizer
    #Créer le collage
    #Retourner le mask et le collage
    pass