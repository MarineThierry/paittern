import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def my_unet(starting_power, input_size):
    inputs = Input(input_size)
    factor = 2 ** starting_power
    
    conv1 = Conv2D(factor, kernel_size = (3,3), activation = 'relu', padding = 'same')(inputs)
    conv2 = Conv2D(factor, kernel_size = (3,3), activation = 'relu', padding = 'same')(conv1)
    MaxPool1 = MaxPooling2D(strides = (2,2))(conv2)
    
    conv3 = Conv2D(factor*2, kernel_size = (3,3), activation = 'relu', padding = 'same')(MaxPool1)
    conv4 = Conv2D(factor*2, kernel_size = (3,3), activation = 'relu', padding = 'same')(conv3)
    MaxPool2 = MaxPooling2D(strides = (2,2))(conv4)
    
    conv5 = Conv2D(factor*4, kernel_size = (3,3), activation = 'relu', padding = 'same')(MaxPool2)
    conv6 = Conv2D(factor*4, kernel_size = (3,3), activation = 'relu', padding = 'same')(conv5)
    MaxPool3 = MaxPooling2D(strides = (2,2))(conv6)
    
    conv7 = Conv2D(factor*8, kernel_size = (2,2), activation = 'relu', padding = 'same')(MaxPool3)
    conv8 = Conv2D(factor*8, kernel_size = (2,2), activation = 'relu', padding = 'same')(conv7)
    MaxPool4 = MaxPooling2D(strides = (2,2))(conv8)
    
    latent = Conv2D(factor*16, kernel_size = (2,2), activation = 'relu', padding = 'same')(MaxPool4)
    latent2 = Conv2D(factor*16, kernel_size = (2,2), activation = 'relu', padding = 'same')(latent)
    
    up1 = UpSampling2D(size = (2,2))(latent2)
    conc1 = concatenate([up1, conv8])
    t_conv1 = Conv2DTranspose(factor*16, kernel_size = (2,2), activation = 'relu', padding = 'same')(conc1)
    t_conv2 = Conv2DTranspose(factor*16, kernel_size = (2,2), activation = 'relu', padding = 'same')(t_conv1)
    
    up2 = UpSampling2D(size = (2,2))(t_conv2)
    conc2 = concatenate([up2, conv6])
    t_conv3 = Conv2DTranspose(factor*8, kernel_size = (3,3), activation = 'relu', padding = 'same')(conc2)
    t_conv4 = Conv2DTranspose(factor*8, kernel_size = (3,3), activation = 'relu', padding = 'same')(t_conv3)
    
    up3 = UpSampling2D(size = (2,2))(t_conv4)
    conc3 = concatenate([up3, conv4])
    t_conv5 = Conv2DTranspose(factor*4, kernel_size = (3,3), activation = 'relu', padding = 'same')(conc3)
    t_conv6 = Conv2DTranspose(factor*4, kernel_size = (3,3), activation = 'relu', padding = 'same')(t_conv5)
    
    up4 = UpSampling2D(size = (2,2))(t_conv6)
    conc4 = concatenate([up4, conv2])
    t_conv7 = Conv2DTranspose(factor*2, kernel_size = (3,3), activation = 'relu', padding = 'same')(conc4)
    t_conv8 = Conv2DTranspose(factor*2, kernel_size = (3,3), activation = 'relu', padding = 'same')(t_conv7)
    
    out = Conv2D(1, 1, activation = 'sigmoid')(t_conv8)
    
    model = Model(inputs = inputs, outputs = out)
    
    return model