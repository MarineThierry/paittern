import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, GridDistortion, OpticalDistortion, ChannelShuffle, CoarseDropout, CenterCrop, Crop, Rotate

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    """ Loading the images and masks """
    X = sorted(glob(os.path.join(path, "images", "*.jpg")))
    return X

def augment_data(images, masks, save_path, augment=True):
    H = 288
    W = 288
    print(H)
    for x, y in tqdm(zip(images, masks), total=len(images)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading the image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        dim1 = x.shape[0]
        dim2 = x.shape[1]

        x = cv2.resize(x, (int(round(dim2*0.5)), int(round(dim1*0.5))))
        y = cv2.resize(y, (int(round(dim2*0.5)), int(round(dim1*0.5))))
        
        """ Augmentation """
        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = Rotate(limit=-45, p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]
            

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented["image"]
            y5 = augmented["mask"]

            X = [x, x1, x2, x5]
            Y = [y, y1, y2, y5]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            try:
                """ Center Cropping """
                aug = CenterCrop(H, W, p=1.0)
                augmented = aug(image=i, mask=m)
                i = augmented["image"]
                m = augmented["mask"]

            except Exception as e:
                i = cv2.resize(i, (W, H))
                m = cv2.resize(m, (W, H))

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)
 
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1


if __name__ == "__main__":
    
    """ Load the dataset """
    data_path = "people_segmentation"
    X = sorted(glob(os.path.join(data_path, "images", "*.jpg")))
    Y = sorted(glob(os.path.join(data_path, "masks", "*.png")))
    
    """ Create directories to save the augmented data """
    create_dir("new_data/image/")
    create_dir("new_data/mask/")

    """ Data augmentation """
    augment_data(X, Y, "new_data/", augment=True)