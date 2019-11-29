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
from keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config

config.num_epochs = 30
config.batch_size = 2
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256


BATCH_SIZE = 2
val_dir = 'data/test'
train_dir = 'data/train'

train_in_images = glob.glob("data/train/*-in.jpg")
train_out_images = glob.glob("data/train/*-out.jpg")

valid_in_images = glob.glob("data/test/*-in.jpg")
valid_out_images = glob.glob("data/test/*-out.jpg")

#print (train_in_images[:5])
#print (train_out_images[:5])

# automatically get the data if it doesn't exist
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xz -C data", shell=True)

steps_per_epoch = len(
    glob.glob(train_dir + "/*-in.jpg")) // BATCH_SIZE
val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) // BATCH_SIZE


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

'''
def image_generator(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        random.shuffle(input_filenames)
        if counter+batch_size >= len(input_filenames):
            counter = 0
        for i in range(batch_size):
            img = input_filenames[counter + i]
            small_images[i] = np.array(Image.open(img)) / 255.0
            large_images[i] = np.array(
                Image.open(img.replace("-in.jpg", "-out.jpg"))) / 255.0
        yield (small_images, large_images)
        counter += batch_size
'''

def image_generator(batch_size=BATCH_SIZE, mode='train'):
    
    while True:
        
        X = []
        y = []
        
        for _ in range(batch_size):
            
            if mode == 'train':
                idx = random.choice([i for i in range(len(train_in_images))])
                
                img_in = cv2.imread(train_in_images[idx])
                img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
            
                for s in train_out_images:
                    if s[11:-8]==train_in_images[idx][11:-7]:
                        break
                        
                img_out = cv2.imread(s)
                img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

                img_in, img_out = _get_augmented_image(img_in, img_out)
            
            else:
                idx = random.choice([i for i in range(len(valid_in_images))])
				
                img_in = cv2.imread(valid_in_images[idx])
                img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

                for s in valid_out_images:
                    if s[11:-8]==valid_in_images[idx][11:-7]:
                        break
                
                img_out = cv2.imread(s)
                img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

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
                    epochs=30, callbacks=[
                    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.000001), ModelCheckpoint('weights.{epoch:05d}-{val_loss:.2f}.hdf5', verbose=1, save_weights_only=True, period=5), WandbCallback()],
                    validation_steps=val_steps_per_epoch,
                    validation_data=val_generator)