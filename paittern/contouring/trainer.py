from glob import glob
from google.cloud import storage
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from model import deeplabv3_plus
from metrics import dice_loss, dice_coef, iou
import joblib

# project id - replace with your GCP project id
PROJECT_ID='wagon-bootcamp-336718'

# bucket name - replace with your GCP bucket name
BUCKET_NAME='wagon-bootcamp-paittern'
BUCKET_FOLDER='people_segmentation'
BUCKET_FOLDER2='augmented_data'
LOCAL_PATH = '/Users/humbert/Documents/Human-Image-Segmentation-with-DeepLabV3Plus-in-TensorFlow-main/people_segmentation/*'
LOCAL_PATH2 = '/Users/humbert/Documents/Human-Image-Segmentation-with-DeepLabV3Plus-in-TensorFlow-main/new_data/*'

H = 512
W = 512

def get_data(objet, bucket_name):
    """Returns the blobs for images and masks"""

    client = storage.Client()
    blobs = client.list_blobs(bucket_name)

    data = [blob.name for blob in blobs if str(blob.name).startswith(f"{BUCKET_FOLDER2}/train/{objet}")]

    return data

def load_data(data, nb, bucket_name):
    bucket = storage.Client().get_bucket(bucket_name)
    my_list = []
    for name in data[:nb]:
        blob = bucket.blob(name)
        obj = np.array(
        cv2.imdecode(
            np.asarray(bytearray(blob.download_as_string()), dtype=np.uint8), 3
        ))
        my_list.append(obj)
    return np.array(my_list)
    
def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset

if __name__ == "__main__":
    print("Chargement des données")
    X_train = get_data('image', BUCKET_NAME)
    Y_train = get_data('mask', BUCKET_NAME) 
    X_train = load_data(X_train, 20, BUCKET_NAME)
    y_train = load_data(Y_train, 20, BUCKET_NAME)
    
    print("Création d'un validation set")
    X_val = [X_train[-4:]]
    y_val = [y_train[-4:]]
    
    X_train = [X_train[:-4]]
    y_train = [y_train[:-4]]
    
    """ Hyperparameters """
    print("Assignation des hyper-paramètres")
    batch_size = 2
    lr = 1e-4
    num_epochs = 20
    
    """ Datasets """
    print("Création d'un dataset train et d'un ataset val")
    train_dataset = tf_dataset(X_train, y_train, batch=batch_size)
    valid_dataset = tf_dataset(X_val, y_val, batch=batch_size)
    
    """ Model """
    print("Création du modèle")
    model = deeplabv3_plus((H, W, 3))
    print("Modèle en cours de compilation")
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[Recall(), Precision()])
    
    print("Création des callbacks")
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]

    print("Entrainement du modèle")
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
    
    print("Sauvegarde du modèle")
    joblib.dump(model, "contouring.joblib")
    
    