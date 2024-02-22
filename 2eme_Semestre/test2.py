# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:19:34 2023

@author: Guilem
"""

import cv2, os
import numpy as np
from os import path
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from keras import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import matplotlib.pyplot as plt

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

def create_model(activation):
    
    model = Sequential() #add model layers

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(300, 300, 1)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(units=50, activation=activation)) 

    model.add(Dense(112, activation='softmax'))
    
    return model

def save_loss(history, nb_cc_value, ethnie, proportion):
    d = history.history
    fig=plt.figure()
    plt.plot(range(epoques), d.get('loss'))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    fig.savefig(fname= 'C:/Users/Guilem/Documents/GitHub/Stage-L3-CNN-ORE/2eme_Semestre/ID_Results/'+str(nb_cc_value)+"_"+ethnie+"_"+str(proportion)+"_"+'.png')

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
    
    return xtrain, ytrain, name_train, datagen.flow(xtrain, ytrain, batch_size=64)

# %%--------------------------------------------------Initialization
dirname= path.dirname('C:/Users/Guilem/Documents/GitHub/Stage-L3-CNN-ORE/2eme_Semestre')
IM_DIR= path.join('C:/Users/Guilem/Documents/GitHub/Stage-L3-CNN-ORE/STIM_NB_LumNorm')

# construct the argument parser and parse the arguments
args = globals().get('args', None)
nb_cc_value = args.hidden
ethnie = args.ethnie
proportion = args.proportion
epoques = args.epoques
print(nb_cc_value, ethnie, proportion)

#create data from folder
data, nb_reduit, nb_im = create_data(IM_DIR, ethnie, proportion)
os.chdir(dirname)

#Processing data
xtrain, ytrain, name_train, train_iterator = processing_data(data, nb_im)

#%%----------------------------------------TRAIN 
# create model

model = KerasClassifier(model=create_model, epochs=10, batch_size=10)
# define the grid search parameters
activation = ['relu', 'tanh', 'sigmoid'] #'softmax', 'softplus', 'softsign', 'hard_sigmoid', 'linear', 
param_grid = dict(model__activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(xtrain, ytrain)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))