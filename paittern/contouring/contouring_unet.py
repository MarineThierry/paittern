import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def my_unet(starting_power, input_size):
    inputs = Input(input_size)
    norm_inputs = Rescaling(1/255)(inputs)
    factor = 2 ** starting_power
    
    conv1 = Conv2D(factor, kernel_size = (3,3), activation = 'relu', padding = 'same')(norm_inputs)
    conv2 = Conv2D(factor, kernel_size = (3,3), activation = 'relu', padding = 'same')(conv1)
    #conv2 = BatchNormalization()(conv2)
    MaxPool1 = MaxPooling2D(strides = (2,2))(conv2)
    
    conv3 = Conv2D(factor*3, kernel_size = (3,3), activation = 'relu', padding = 'same')(MaxPool1)
    conv4 = Conv2D(factor*3, kernel_size = (3,3), activation = 'relu', padding = 'same')(conv3)
    conv4 = BatchNormalization()(conv4)
    MaxPool2 = MaxPooling2D(strides = (2,2))(conv4)
    
    conv5 = Conv2D(factor*6, kernel_size = (3,3), activation = 'relu', padding = 'same')(MaxPool2)
    conv6 = Conv2D(factor*6, kernel_size = (3,3), activation = 'relu', padding = 'same')(conv5)
    #conv6 = BatchNormalization()(conv6)
    MaxPool3 = MaxPooling2D(strides = (2,2))(conv6)
    
    conv7 = Conv2D(factor*12, kernel_size = (2,2), activation = 'relu', padding = 'same')(MaxPool3)
    conv8 = Conv2D(factor*12, kernel_size = (2,2), activation = 'relu', padding = 'same')(conv7)
    #conv8 = Dropout(0.5)(conv8)
    conv8 = BatchNormalization()(conv8)
    MaxPool4 = MaxPooling2D(strides = (2,2))(conv8)
    
    latent = Conv2D(factor*24, kernel_size = (2,2), activation = 'relu', padding = 'same')(MaxPool4)
    latent2 = Conv2D(factor*24, kernel_size = (2,2), activation = 'relu', padding = 'same')(latent)
    #latent2 = Dropout(0.5)(latent2)
    #latent2 = BatchNormalization()(latent2)
    
    up1 = UpSampling2D(size = (2,2))(latent2)
    conc1 = concatenate([up1, conv8])
    t_conv1 = Conv2D(factor*16, kernel_size = (2,2), activation = 'relu', padding = 'same')(conc1)
    t_conv2 = Conv2D(factor*16, kernel_size = (2,2), activation = 'relu', padding = 'same')(t_conv1)
    #t_conv2 = BatchNormalization()(t_conv2)
    
    up2 = UpSampling2D(size = (2,2))(t_conv2)
    conc2 = concatenate([up2, conv6])
    t_conv3 = Conv2D(factor*8, kernel_size = (3,3), activation = 'relu', padding = 'same')(conc2)
    t_conv4 = Conv2D(factor*8, kernel_size = (3,3), activation = 'relu', padding = 'same')(t_conv3)
    #t_conv4 = BatchNormalization()(t_conv4)
    
    up3 = UpSampling2D(size = (2,2))(t_conv4)
    conc3 = concatenate([up3, conv4])
    t_conv5 = Conv2D(factor*4, kernel_size = (3,3), activation = 'relu', padding = 'same')(conc3)
    t_conv6 = Conv2D(factor*4, kernel_size = (3,3), activation = 'relu', padding = 'same')(t_conv5)
    t_conv6 = BatchNormalization()(t_conv6)
    
    up4 = UpSampling2D(size = (2,2))(t_conv6)
    conc4 = concatenate([up4, conv2])
    t_conv7 = Conv2D(factor*2, kernel_size = (3,3), activation = 'relu', padding = 'same')(conc4)
    t_conv8 = Conv2D(factor*2, kernel_size = (3,3), activation = 'relu', padding = 'same')(t_conv7)
    #t_conv8 = BatchNormalization()(t_conv8)
    t_conv8 = Dropout(0.2)(t_conv8)
    
    out = Conv2D(1, 1, activation = 'sigmoid')(t_conv8)
    
    model = Model(inputs = inputs, outputs = out)
    
    return model