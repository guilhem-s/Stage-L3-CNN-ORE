# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:19:34 2023

@author: Guilem
"""

import cv2, os
import tensorflow as tf
import numpy as np
from os import path
from pathlib import Path
import keras
from keras import Sequential
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from scipy.spatial import distance
from keras.utils import to_categorical
import matplotlib.pyplot as plt
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

args = globals().get('args', None)
nb_cc_value = args.hidden
ethnie = args.ethnie
proportion = args.proportion
epoques = args.epoques
#batch = []

def create_data(IM_DIR, ethnie, proportion): 
    # takes images from folder and associates them with a label - create a structure with names - labels - im
    prop = {0: [56, 56], 10: [50, 62], 20: [42, 70], 30: [32, 80], 40: [19, 93], 50: [0, 112]}
    os.chdir(IM_DIR)
    data = []
    for imname in os.listdir(IM_DIR):
        data.append([np.array(cv2.imread(imname, cv2.IMREAD_GRAYSCALE)), imname])
    np.random.shuffle(data)
    # Reduit le nombre de visages dans le training set
    #int(56*proportion / (100-proportion)) calcul du nb d'images à conserver
    a_enlever, cpt_final = prop[proportion][0], prop[proportion][1]   #56 - nb d'images à conserver
    i=0
    while a_enlever>0:
        if ethnie in data[i][1]:
            a_enlever-=1
            data.pop(i)
        else:
            i+=1
    return data, 56 - a_enlever, cpt_final

def build_model(learning_rate = 0.001,momentum=0.9):
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Conv2D(6, kernel_size=7, activation='relu'))
    model.add(keras.layers.MaxPooling2D((3,3), strides=2))
    model.add(keras.layers.Conv2D(16, kernel_size=5, activation='relu'))
    model.add(keras.layers.MaxPooling2D((3,3)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(nb_cc_value, activation='relu'))
    model.add(keras.layers.Dense(nb_im, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
                  loss="mse", metrics=tf.keras.metrics.MeanAbsoluteError())
    return model

def processing_data(data, nb_im):
    pairs_train=[]
    name_train=[]

    for i in range(0,len(data)):
            pairs_train.append(data[i][0]) # image sous forme de matrice
            name_train.append(data[i][1])  # nom de l'image
            
    xtrain=np.array(pairs_train).astype(int)
    ytrain = to_categorical(np.arange(nb_im)) # one hot encode target values

    # reshape dataset to have a single channel
    width = xtrain.shape[1] # dimension de l'image
    xtrain = xtrain.reshape((nb_im, width, width, 1))

    # create generator (1.0/255.0 = 0.003921568627451)
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    return xtrain, ytrain, name_train, datagen.flow(xtrain, ytrain)

dirname= path.dirname('C:/Users/Guilem/Documents/GitHub/Stage-L3-CNN-ORE/2eme_Semestre')
IM_DIR= path.join('C:/Users/Guilem/Documents/GitHub/Stage-L3-CNN-ORE/STIM_NB_LumNorm')

# construct the argument parser and parse the arguments
print(nb_cc_value, ethnie, proportion)
data, nb_reduit, nb_im = create_data(IM_DIR, ethnie, proportion)

#Processing data
xtrain, ytrain, name_train, train_iterator = processing_data(data, nb_im)
for i in range(nb_im//32):
    x, y = train_iterator.next()

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
from sklearn.model_selection import GridSearchCV
param_distribs = {"learning_rate": [2, 1.5, 1, 0.95, 0.9, 0.85, 0.8],
                  "momentum": [0.75,0.76,0.77,0.78]}
grid_search_cv = GridSearchCV(keras_reg, param_distribs, cv=2)
grid_search_cv.fit(x, y, batch_size=32, epochs=epoques, shuffle=True)

grid_search_cv.best_params_
tostr = grid_search_cv.best_params_
tostr = str(tostr)
print(tostr)
with open(Path('C:/Users/Guilem/Documents/GitHub/Stage-L3-CNN-ORE/2eme_Semestre/ID_Results/GRID2.txt'), 'a') as output:
    output.write(f"Best parameters: {tostr}")