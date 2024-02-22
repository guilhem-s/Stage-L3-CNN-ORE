# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:19:34 2023

@author: Guilem
"""

import cv2, os
import tensorflow
import numpy as np
from os import path
from keras import Sequential
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import matplotlib.pyplot as plt
config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.compat.v1.Session(config=config)
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

def create_model(nb_cc_value, nb_im, activation_function):
    
    model = Sequential() #add model layers

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(300, 300, 1)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(units=nb_cc_value, activation=activation_function)) 

    model.add(Dense(nb_im, activation='sigmoid'))
    
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

def grid_search(xtrain, ytrain, nb_cc_value, nb_im, opt, batch_size, epochs, activation_functions):
    best_accuracy = 0
    params = ""
    for batch in batch_size:
        for epoques in epochs:
            for activation_function in activation_functions:
                print(batch, " ", activation_function)
                model = create_model(nb_cc_value, nb_im, activation_function)

                model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
                history = model.fit(xtrain, ytrain, epochs=epoques, batch_size=batch, verbose=3)

                accuracy = np.max(history.history['accuracy'])
                result = f"{accuracy} pour les paramètres : fon={activation_function}, epoques={epoques}, batch_size={batch}"
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = result

                params += result+'\n'
    return best_params, params
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
"""learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]"""

# Définir les valeurs à tester
activation_functions = ['relu', 'sigmoid', 'tanh']
batch_size = [10, 20, 30, 40]
epoques = [20, 50, 100, 150]
opt = Adam()              
"""opt2 = SGD(learning_rate, momentum) # descente de gradient""" 

# Appeler la fonction de recherche sur grille
params, best_params = grid_search(xtrain, ytrain, nb_cc_value, nb_im, opt, batch_size, epochs=epoques, activation_functions=activation_functions)

#plot and save training curve
#save_loss(history, nb_cc_value, ethnie, proportion)

#%%----------------------------------------TEST 

# summarize results
print(best_params)
print(params)