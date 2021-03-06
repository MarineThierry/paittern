from google.cloud import storage
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU, BinaryIoU
from tensorflow.keras.losses import BinaryCrossentropy
from paittern.contouring.contouring_unet import my_unet
#from paittern.contouring.brouillons.model import deeplabv3_plus
from sklearn.utils import shuffle
import os

# project id - replace with your GCP project id
PROJECT_ID='wagon-bootcamp-336718'

# bucket name - replace with your GCP bucket name
BUCKET_NAME='wagon-bootcamp-paittern'
BUCKET_FOLDER='people_segmentation'
BUCKET_FOLDER2='augmented_data'
BUCKET_FOLDER3='new_data'
LOCAL_PATH = '/Users/humbert/Documents/Human-Image-Segmentation-with-DeepLabV3Plus-in-TensorFlow-main/people_segmentation/*'
LOCAL_PATH2 = '/Users/humbert/Documents/Human-Image-Segmentation-with-DeepLabV3Plus-in-TensorFlow-main/new_data/*'

H = 288
W = 288

def get_data(objet, bucket_name):
    """Returns the blobs for images and masks"""

    client = storage.Client()
    blobs = client.list_blobs(bucket_name)

    data = [blob.name for blob in blobs if str(blob.name).startswith(f"{BUCKET_FOLDER3}/{BUCKET_FOLDER3}/{objet}")]
    return data

def load_data(data, nb, bucket_name, dim):
    bucket = storage.Client().get_bucket(bucket_name)
    my_list = []
    for name in data[:nb]:
        blob = bucket.blob(name)
        obj = np.array(
        cv2.imdecode(
            np.asarray(bytearray(blob.download_as_string()), dtype=np.uint8), dim
        ))
        my_list.append(obj)
    return np.array(my_list)
    
def read_image(x):
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(x):
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

STORAGE_LOCATION = 'models/contouring/contouring'

def upload_model_to_gcp(model_name):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    # image log
    #print('uploading image_log to gcp')
    #blob = bucket.blob(f'{STORAGE_LOCATION}/{model_name}.pickle')
    #blob.upload_from_filename(f'./image_logs/{model_name}-img_log.pickle')
    # model
    print('uploading model to gcp')
    current_wd = os.getcwd()
    for root, directories, files in os.walk('./saved_models'):
        for name in files:
            full_name = os.path.join(root.replace(current_wd,""), name)
            print('uploading :',full_name)
            blob = bucket.blob(f'{STORAGE_LOCATION}/{full_name.strip("./saved_models/")}')
            blob.upload_from_filename(f'{full_name}')
    print('upload finished\n')
    print('all done')


def save_model(model, model_name):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    model.save(f"./saved_models/{model_name}")
    print("saved contouring locally")

    # Implement here
    upload_model_to_gcp(model_name)
    print(f"uploaded contouring.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, "float32")
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

if __name__ == "__main__":    
    # Model
    print("Cr??ation du mod??le")
    model = my_unet(3, (288,288,3))
    model.summary()
    #print("Cr??ation metrics")
    my_iou = BinaryIoU(name = "my_iou", target_class_ids=[1])
    
    print("Cr??ation des callbacks")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ]
    # Hyperparameters 
    print("Assignation des hyper-param??tres")
    batch_size = 32
    lr = 0.001
    num_epochs = 200
    
    print("Mod??le en cours de compilation")
    model.compile(loss=BinaryCrossentropy(from_logits=True), optimizer=Adam(lr), metrics = ["accuracy", my_iou])
    
    
    print("Chargement des donn??es")
    X_train = get_data('image', BUCKET_NAME)
    Y_train = get_data('mask', BUCKET_NAME) 
    print("get_data r??alis??")
    n_test = 2000
    random_int = 42
    n = 18000
    X_train = X_train[:-n_test]
    Y_train = Y_train[:-n_test]
    X_train = shuffle(X_train, random_state=random_int)
    Y_train = shuffle(Y_train, random_state=random_int)
    X_train = load_data(X_train, n, BUCKET_NAME, 3)
    y_train = load_data(Y_train, n, BUCKET_NAME, 0)
    print(X_train.shape, y_train.shape)

    
    #print("Cr??ation d'un validation set")
    #X_val = X_train[round(0.8*n):]
    #y_val = y_train[round(0.8*n):]
    
    #X_train = X_train[:round(0.8*n)]
    #y_train = y_train[:round(0.8*n)]
    
    # Datasets 
    #print("Cr??ation d'un dataset train et d'un dataset val")
    #train_dataset = tf_dataset(X_train, y_train, batch=batch_size)
    #valid_dataset = tf_dataset(X_val, y_val, batch=batch_size)

    print("Entrainement du mod??le")
    model.fit(
       X_train, y_train,
       epochs=num_epochs,
       batch_size = batch_size,
       validation_split=0.2,
       callbacks=callbacks,
       workers = -1
    )
    
    print("Sauvegarde du mod??le")
    save_model(model, 'contouring_300ep_bc_logits_mu3')
