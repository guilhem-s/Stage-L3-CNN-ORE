import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint

def create_autoencoder(input_latent, shape):
    input_img = tf.keras.Input(shape=shape)
    print("Shape de l input = ", input_img.shape)
    # Feature extractor (encoder)
    x = tf.keras.layers.BatchNormalization()(input_img)
    x = tf.keras.layers.Conv2D(8, kernel_size=11, padding='same', activation='relu', use_bias=False)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(16, kernel_size=7, activation='relu',padding='same', use_bias=False)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=5, activation='relu',padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu',padding='same', use_bias=False)(x)
    #x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu',padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu',padding='same', use_bias=False)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)  
    x = tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu',padding='same', use_bias=False)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)    
    x = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu',padding='same', use_bias=False)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=3, activation='relu',padding='same', use_bias=False)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(1024, kernel_size=3, activation='relu',padding='same', use_bias=False)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    
    # Espace latent
    encoded = x
    latent_space_layer = tf.keras.layers.Dense(units=1*1*input_latent)(encoded)
    latent_space_layer_norm = tf.keras.layers.BatchNormalization(name='latent_space_norm')(latent_space_layer)
    
    # Feature redear (decoder)
    x_recon = tf.keras.layers.Reshape(target_shape=(1, 1, input_latent))(latent_space_layer_norm)
    print("Shape de l x recon = ", x_recon.shape)
    x_recon = tf.keras.layers.Conv2DTranspose(1024, kernel_size=5, activation='relu', padding='same', use_bias=False)(x_recon)
    x_recon = tf.keras.layers.BatchNormalization()(x_recon)
    x_recon = tf.keras.layers.Conv2DTranspose(524, kernel_size=5, activation='relu', strides=(2, 2), padding='same', use_bias=False)(x_recon)
    x_recon = tf.keras.layers.BatchNormalization()(x_recon)
    x_recon = tf.keras.layers.Conv2DTranspose(256, kernel_size=5, activation='relu', strides=(2, 2), padding='same', use_bias=False)(x_recon)
    x_recon = tf.keras.layers.BatchNormalization()(x_recon)
    x_recon = tf.keras.layers.Conv2DTranspose(128, kernel_size=5, activation='relu', strides=(2, 2), padding='same', use_bias=False)(x_recon)
    x_recon = tf.keras.layers.BatchNormalization()(x_recon)
    x_recon = tf.keras.layers.Conv2DTranspose(64, kernel_size=5, activation='relu', strides=(2, 2), padding='same', use_bias=False)(x_recon)
    x_recon = tf.keras.layers.BatchNormalization()(x_recon)
    x_recon = tf.keras.layers.Conv2DTranspose(64, kernel_size=5, activation='relu', padding='same', use_bias=False)(x_recon)
    x_recon = tf.keras.layers.BatchNormalization()(x_recon)
    x_recon = tf.keras.layers.Conv2DTranspose(32, kernel_size=5, activation='relu', strides=(2, 2), padding='same', use_bias=False)(x_recon)
    x_recon = tf.keras.layers.BatchNormalization()(x_recon)
    x_recon = tf.keras.layers.Conv2DTranspose(32, kernel_size=5, activation='relu', padding='same', use_bias=False)(x_recon)
    x_recon = tf.keras.layers.BatchNormalization()(x_recon)
    x_recon = tf.keras.layers.Conv2DTranspose(16, kernel_size=5, activation='relu', strides=(2, 2), padding='same', use_bias=False)(x_recon)
    x_recon = tf.keras.layers.BatchNormalization()(x_recon)
    x_recon = tf.keras.layers.Conv2DTranspose(8, kernel_size=5, activation='relu', strides=(2, 2), padding='same', use_bias=False)(x_recon)
    x_recon = tf.keras.layers.BatchNormalization()(x_recon)
    x_recon = tf.keras.layers.Conv2DTranspose(1, kernel_size=11, activation='sigmoid', padding='same')(x_recon)  
    x_recon = tf.keras.layers.Resizing(224, 224)(x_recon)
    print("Shape de l x recon 2 = ", x_recon.shape)
    # Full model
    model = tf.keras.Model(inputs=input_img, outputs=[x_recon])
    
    return model

train_datagen = ImageDataGenerator(rescale=1./255)
dataset_white = train_datagen.flow_from_directory(
    '/silenus/PROJECTS/pr-deepneuro/simeaog/Datasets_FairFace/train',
    classes=['White'],
    target_size=(150, 150), 
    batch_size=32, 
    class_mode='input',
    color_mode='grayscale')

shape = (150, 150,1)
input_latent = 128 #nombre de dimensions de l'espace latent, celui dont on veut qu'il modélise le face space de Tim Valentine donc
model = create_autoencoder(input_latent, shape)
chekpoint_path = 'fairface_autoencoder_uni.hdf5'
checkpoint_callback = ModelCheckpoint(
    filepath=chekpoint_path,
    save_weights_only=False,
    save_frequency='epoch',
    save_best_only=True
    )

model.compile(optimizer='SGD', loss='mse', metrics=['mae'])

history = model.fit(x=dataset_white, epochs=30, batch_size=16, callbacks=[checkpoint_callback])
