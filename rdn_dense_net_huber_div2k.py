import random
import cv2
import glob
import subprocess
import os
from PIL import Image
import numpy as np
#from tensorflow.keras.models import Sequential
#from tensorflow.keras import layers
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Activation, Dense, Add, UpSampling2D, Concatenate, Layer, Dropout, Cropping2D, Lambda
from keras.callbacks import Callback, ReduceLROnPlateau
from keras.optimizers import Adam
import tensorflow as tf


BATCH_SIZE = 2
valid_dir = 'DIV2K_valid_HR/*'
train_dir = 'DIV2K_train_HR/*'

train_images = glob.glob(train_dir)
valid_images = glob.glob(valid_dir)

steps_per_epoch = len(
    train_images) // BATCH_SIZE
val_steps_per_epoch = len(
    valid_images) // BATCH_SIZE


def _get_augmented_image(img_new, img):
    
    gamma = random.choice([i for i in np.arange(0.5, 2.5, 0.25)])
   
    img_new = img_new / 255.
    img_new = img_new ** (1/gamma)
    img_new *= 255
    img_new = img_new.astype('uint8')
	
    img = img / 255.
    img = img ** (1/gamma)
    img *= 255
    img = img.astype('uint8')

    augmentation = np.random.choice(['Flip', 'Rotation', 'None'], p=[0.2, 0.6, 0.2])

    if augmentation == 'Flip':
        img_new = np.fliplr(img_new)
        img = np.fliplr(img)
    
    elif augmentation == 'Rotation':
        angle = random.choice([i for i in range(-10, 10, 2)])

        rows, cols, channels = img_new.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        img_new = cv2.warpAffine(img_new, M, (cols, rows))

        rows, cols, channels = img.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        img = cv2.warpAffine(img, M, (cols, rows))

    else:
        img_new = img_new
        img = img
		
    return img_new, img


def image_generator(batch_size=BATCH_SIZE, mode='train'):
    
    while True:
        
        X = []
        y = []
        
        for _ in range(batch_size):
            
            if mode == 'train':
                img_fi = random.choice(train_images)
                
                img = cv2.imread(img_fi)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                shape = img.shape
                row = random.randint(0, shape[0]-256)
                col = random.randint(0, shape[1]-256)
                img_out = img[row:row+256, col:col+256, :]

                img_in = cv2.resize(img_out, (32, 32), interpolation=cv2.INTER_CUBIC)

                img_in, img_out = _get_augmented_image(img_in, img_out)
            
            else:
                img_fi = random.choice(valid_images)
                
                img = cv2.imread(img_fi)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                shape = img.shape
                row = random.randint(0, shape[0]-256)
                col = random.randint(0, shape[1]-256)
                img_out = img[row:row+256, col:col+256, :]

                img_in = cv2.resize(img_out, (32, 32), interpolation=cv2.INTER_CUBIC)


            img_in = img_in / 255.
            img_out = img_out / 255.
            
            X.append(img_in)
            y.append(img_out)
            
        X = np.asarray(X)
        y = np.asarray(y)
        
        yield X, y


def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

def image_psnr(y_true, y_pred):
	
	return tf.image.psnr(y_true, y_pred, max_val=1.0)


train_generator = image_generator(BATCH_SIZE, 'train')
val_generator = image_generator(BATCH_SIZE, 'valid')
#in_sample_images, out_sample_images = next(val_generator)

'''
class ImageLogger(Callback):
    def on_epoch_end(self, epoch, logs):
        preds = self.model.predict(in_sample_images)
        in_resized = []
        for arr in in_sample_images:
            # Simple upsampling
            in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
        wandb.log({
            "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for i, o in enumerate(preds)]
        }, commit=False)
'''

def residual_dense_block(inp_layer, growth_factor=64):
	
    x = Conv2D(growth_factor, kernel_size=(3, 3), padding='same')(inp_layer)
    x = Activation('relu')(x)
	
    cat_1 = Concatenate()([inp_layer, x])
	
    y = Conv2D(growth_factor, kernel_size=(3, 3), padding='same')(cat_1)
    y = Activation('relu')(y)
	
    cat_2 = Concatenate()([cat_1, y])
	
    z = Conv2D(growth_factor, kernel_size=(3, 3), padding='same')(cat_2)
    z = Activation('relu')(z)

    cat_3 = Concatenate()([cat_2, z])
	
    x1 = Conv2D(growth_factor, kernel_size=(3, 3), padding='same')(cat_3)
    x1 = Activation('relu')(x1)
	
    cat_4 = Concatenate()([cat_3, x1])
	
    y1 = Conv2D(growth_factor, kernel_size=(3, 3), padding='same')(cat_4)
    y1 = Activation('relu')(y1)
	
    cat_5 = Concatenate()([cat_4, y1])

    z1 = Conv2D(growth_factor, kernel_size=(3, 3), padding='same')(cat_5)
    z1 = Activation('relu')(z1)
	
    cat_6 = Concatenate()([cat_5, z1])

    conv_final = Conv2D(growth_factor, kernel_size=(1, 1), padding='same')(cat_6)
	
    add = Add()([inp_layer, conv_final])
    return add


inp = Input((32, 32, 3))

conv_1 = Conv2D(64, kernel_size=(3, 3), padding='same')(inp)
conv_1 = Activation('relu')(conv_1)

conv_2 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv_1)
conv_2 = Activation('relu')(conv_2)

rdb_1 = residual_dense_block(conv_2)
rdb_2 = residual_dense_block(rdb_1)
rdb_3 = residual_dense_block(rdb_2)
rdb_4 = residual_dense_block(rdb_3)
rdb_5 = residual_dense_block(rdb_4)

cat_1 = Concatenate()([rdb_1, rdb_2, rdb_3, rdb_4, rdb_5])
conv_3 = Conv2D(64, kernel_size=(1, 1), padding='same')(cat_1)
conv_4 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv_3)

add_1 = Add()([conv_1, conv_4])
up_1 = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(add_1)

conv_5 = Conv2D(64, kernel_size=(3, 3), padding='same')(up_1)
conv_5 = Activation('relu')(conv_5)

conv_6 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv_5)
conv_6 = Activation('relu')(conv_6)

rdb_6 = residual_dense_block(conv_6)
rdb_7 = residual_dense_block(rdb_6)
rdb_8 = residual_dense_block(rdb_7)
rdb_9 = residual_dense_block(rdb_8)
rdb_10 = residual_dense_block(rdb_9)

cat_2 = Concatenate()([rdb_6, rdb_7, rdb_8, rdb_9, rdb_10])
conv_7 = Conv2D(64, kernel_size=(1, 1), padding='same')(cat_2)
conv_8 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv_7)

add_2 = Add()([conv_5, conv_8])
up_2 = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(add_2)

conv_9 = Conv2D(64, kernel_size=(3, 3), padding='same')(up_2)
conv_9 = Activation('relu')(conv_9)

conv_10 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv_9)
conv_10 = Activation('relu')(conv_10)

rdb_11 = residual_dense_block(conv_10, growth_factor=64)
rdb_12 = residual_dense_block(rdb_11, growth_factor=64)
rdb_13 = residual_dense_block(rdb_12, growth_factor=64)
rdb_14 = residual_dense_block(rdb_13, growth_factor=64)
rdb_15 = residual_dense_block(rdb_14, growth_factor=64)

cat_3 = Concatenate()([rdb_11, rdb_12, rdb_13, rdb_14, rdb_15])
conv_11 = Conv2D(64, kernel_size=(1, 1), padding='same')(cat_3)
conv_12 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv_11)

add_3 = Add()([conv_9, conv_12])
up_3 = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(add_3)

conv_13 = Conv2D(64, kernel_size=(3, 3), padding='same')(up_3)
conv_13 = Activation('relu')(conv_13)

conv_14 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv_13)
conv_14 = Activation('relu')(conv_14)

rdb_16 = residual_dense_block(conv_14)
rdb_17 = residual_dense_block(rdb_16)
rdb_18 = residual_dense_block(rdb_17)
rdb_19 = residual_dense_block(rdb_18)
rdb_20 = residual_dense_block(rdb_19)

cat_4 = Concatenate()([rdb_16, rdb_17, rdb_18, rdb_19, rdb_20])
conv_15 = Conv2D(64, kernel_size=(1, 1), padding='same')(cat_4)
conv_16 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv_15)

add_4 = Add()([conv_13, conv_16])

out = Conv2D(3, kernel_size=(3, 3), padding='same')(add_4)

model = Model(inputs=inp, outputs=out)
print (model.summary())

# DONT ALTER metrics=[perceptual_distance]
model.compile(optimizer=Adam(), loss=tf.losses.huber_loss, metrics=[perceptual_distance, image_psnr])

model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=20, callbacks=[
                    ReduceLROnPlateau(factor=0.1, patience=2, min_lr=0.000001)],
                    validation_steps=val_steps_per_epoch,
                    validation_data=val_generator)
					
model.save_weights('huber_rdn20_subpixel_16M_weights.hdf5')
