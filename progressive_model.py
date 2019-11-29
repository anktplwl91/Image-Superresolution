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
from keras.layers import Input, Conv2D, Activation, Dense, Add, UpSampling2D, Concatenate, Layer, Dropout, Cropping2D, Lambda, GaussianNoise
from keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config

config.num_epochs = 100
config.batch_size = 16
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256


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
    glob.glob(train_dir + "/*-in.jpg")) // config.batch_size
val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) // config.batch_size


def _get_augmented_image(img_new, img):
    
    gamma = random.choice([i for i in np.arange(0.5, 2.5, 0.15)])
   
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
        angle = random.choice([i for i in range(-20, 20, 2)])

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
            img_in = np.array(Image.open(img))
            img_out = np.array(Image.open(img.replace("-in.jpg", "-out.jpg")))

            img_in, img_out = _get_augmented_image(img_in, img_out)
            small_images[i] = img_in / 255.
            large_images[i] = img_out / 255.

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
'''

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


train_generator = image_generator(config.batch_size, train_dir)
val_generator = image_generator(config.batch_size, val_dir)
in_sample_images, out_sample_images = next(val_generator)


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


def dense_compression_block(inp_layer, growth_factor):

    def dense_block(l, growth_factor):
	
        l = Conv2D(growth_factor, kernel_size=(1, 1), padding='same')(l)
        l = Activation('relu')(l)
        l = Conv2D(growth_factor, kernel_size=(3, 3), padding='same')(l)
        return l
	
    x = dense_block(inp_layer, growth_factor)
    x = Conv2D(growth_factor, kernel_size=(1, 1), padding='same')(x)
	
    y = dense_block(x, growth_factor)
    y = Conv2D(growth_factor, kernel_size=(1, 1), padding='same')(y)
	
    z = dense_block(y, growth_factor)
    z = Conv2D(growth_factor, kernel_size=(1, 1), padding='same')(z)

    x1 = dense_block(z, growth_factor)
    x1 = Conv2D(growth_factor, kernel_size=(1, 1), padding='same')(x1)
	
    y1 = dense_block(x1, growth_factor)
    y1 = Conv2D(growth_factor, kernel_size=(1, 1), padding='same')(y1)
	
    z1 = dense_block(y1, growth_factor)
    z1 = Conv2D(growth_factor, kernel_size=(1, 1), padding='same')(z1)
	
    add_6 = Add()([inp_layer, z1])

    return add_6


inp = Input((32, 32, 3))
g_inp = GaussianNoise(0.01)(inp)

conv_1 = Conv2D(64, kernel_size=(3, 3), padding='same')(g_inp)
conv_1 = Activation('relu')(conv_1)

conv_2 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv_1)
conv_2 = Activation('relu')(conv_2)

dcb_1 = dense_compression_block(conv_2, growth_factor=64)
dcb_2 = dense_compression_block(dcb_1, growth_factor=64)
dcb_3 = dense_compression_block(dcb_2, growth_factor=64)
dcb_4 = dense_compression_block(dcb_3, growth_factor=64)
dcb_5 = dense_compression_block(dcb_4, growth_factor=64)

conv_3 = Conv2D(64, kernel_size=(3, 3), padding='same')(dcb_5)
conv_3 = Activation('relu')(conv_3)

add_1 = Add()([conv_2, conv_3])
#up_1 = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(add_1)
bi_1 = UpSampling2D(interpolation='bilinear')(add_1)
#up_bi_1 = Add()([up_1, bi_1])

conv_4 = Conv2D(64, kernel_size=(3, 3), padding='same')(bi_1)
conv_4 = Activation('relu')(conv_4)

conv_5 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv_4)
conv_5 = Activation('relu')(conv_5)

dcb_6 = dense_compression_block(conv_5, growth_factor=64)
dcb_7 = dense_compression_block(dcb_6, growth_factor=64)
dcb_8 = dense_compression_block(dcb_7, growth_factor=64)
dcb_9 = dense_compression_block(dcb_8, growth_factor=64)
dcb_10 = dense_compression_block(dcb_9, growth_factor=64)

conv_6 = Conv2D(64, kernel_size=(3, 3), padding='same')(dcb_10)
conv_6 = Activation('relu')(conv_6)

add_2 = Add()([conv_5, conv_6])
#up_2 = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(add_2)
bi_2 = UpSampling2D(interpolation='bilinear')(add_2)
#up_bi_2 = Add()([up_2, bi_2])

conv_6 = Conv2D(64, kernel_size=(3, 3), padding='same')(bi_2)
conv_6 = Activation('relu')(conv_6)

conv_7 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv_6)
conv_7 = Activation('relu')(conv_7)

dcb_11 = dense_compression_block(conv_7, growth_factor=64)
dcb_12 = dense_compression_block(dcb_11, growth_factor=64)
dcb_13 = dense_compression_block(dcb_12, growth_factor=64)
dcb_14 = dense_compression_block(dcb_13, growth_factor=64)
dcb_15 = dense_compression_block(dcb_14, growth_factor=64)

conv_8 = Conv2D(64, kernel_size=(3, 3), padding='same')(dcb_15)
conv_8 = Activation('relu')(conv_8)

add_3 = Add()([conv_7, conv_8])
#up_3 = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(add_3)
bi_3 = UpSampling2D(interpolation='bilinear')(add_3)
#up_bi_3 = Add()([up_3, bi_3])

conv_9 = Conv2D(64, kernel_size=(3, 3), padding='same')(bi_3)
conv_9 = Activation('relu')(conv_9)

conv_10 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv_9)
conv_10 = Activation('relu')(conv_10)

dcb_16 = dense_compression_block(conv_10, growth_factor=64)
dcb_17 = dense_compression_block(dcb_16, growth_factor=64)
dcb_18 = dense_compression_block(dcb_17, growth_factor=64)
dcb_19 = dense_compression_block(dcb_18, growth_factor=64)
dcb_20 = dense_compression_block(dcb_19, growth_factor=64)

conv_11 = Conv2D(64, kernel_size=(3, 3), padding='same')(dcb_20)
conv_11 = Activation('relu')(conv_11)

add_4 = Add()([conv_10, conv_11])

out = Conv2D(3, kernel_size=(3, 3), padding='same')(add_4)

model = Model(inputs=inp, outputs=out)
print (model.summary())

multi_model = multi_gpu_model(model, gpus=2)

# DONT ALTER metrics=[perceptual_distance]
multi_model.compile(optimizer=Adam(lr=0.0001), loss=tf.losses.huber_loss, metrics=[perceptual_distance, image_psnr])

multi_model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=config.num_epochs, callbacks=[
                    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=0.0000001), ModelCheckpoint('dcb20_bilinear_5M_weights.{epoch:10d}-{val_loss:.2f}.hdf5', verbose=1, save_weights_only=True, period=10), WandbCallback(), ImageLogger()],
                    validation_steps=val_steps_per_epoch,
                    validation_data=val_generator)