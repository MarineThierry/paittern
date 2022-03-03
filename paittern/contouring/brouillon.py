from glob import glob
from hashlib import new
from importlib.resources import path
import os
from types import new_class
import numpy as np
import cv2

def create_dir(path):
    """ Test if a file exist and create if no"""
    if os.path.exists(path):
        pass
    else :
        os.mkdir(path)

def get_XY(path_to_file):
    X = sorted(glob(os.path.join(path_to_file,"images","*.jpg")))
    Y = sorted(glob(os.path.join(path_to_file,"masks","*.png")))
    return X, Y
        
def copy_split_data(X, Y, path=os.path.join("..","data"), train_split=0.8):
    """From 2 directories (image and their corresponding mask), create 2 files train and 
    test with X and y inside of each one.
    Finally return X_train, y_train, X_test, y_test """
    create_dir(path)
    create_dir(path+"train")
    create_dir(path+"test")
    create_dir(path+"train/X")
    create_dir(path+"train/Y/")
    create_dir(path+"test/X/")
    create_dir(path+"test/Y/")
    X_train, y_train, X_test, y_test = [],[],[],[]
    n_train = round(len(X)*train_split)
    k = 1
    X = sorted(X)
    Y = sorted(Y)
    for x, y in zip(X[:n_train],Y[:n_train]):
        x = cv2.imread(x)
        X_train.append(x)
        cv2.imwrite(path+"train/X/"+f"img_{k}.png", x)
        y = cv2.imread(y)
        y_train.append(y*255)
        cv2.imwrite(path+"train/Y/"+f"img_{k}.png", y*255)
        k +=1
    for x, y in zip(X[n_train:],Y[n_train:]):
        x = cv2.imread(x)
        X_test.append(x)
        cv2.imwrite(path+"test/X/"+f"img_{k}.png", x)
        y = cv2.imread(y)
        y_test.append(y*255)
        cv2.imwrite(path+"test/Y/"+f"img_{k}.png", y*255)
        k +=1
    return X_train, y_train, X_test, y_test

def resized_images(X, Y, tuple_size):
    X_resized = [cv2.resize(X[k], dsize=tuple_size, 
                          interpolation = cv2.INTER_NEAREST) for k in range(len(X))]
    y_resized = [cv2.resize(Y[k], dsize=tuple_size, 
                          interpolation = cv2.INTER_NEAREST) for k in range(len(Y))]
    return X_resized, y_resized

if __name__ == "__main__":
    X, Y = get_XY()
    print(len(X))
    X_train, y_train = copy_split_data(X[:50], Y[:50], path='data2/')[:2]
    new_X, new_y = resized_images(X_train, y_train, (256,256))
    print(new_X[3].shape)
    print(len(new_X))

