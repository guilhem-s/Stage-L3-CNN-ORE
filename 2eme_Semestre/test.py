# -*- coding: utf-8 -*-
"""
Créé le Mar 12 Dec 18:19:34 2023

@author: Guilem
"""

import cv2
import os
import gc
import csv
import argparse
import numpy as np
from os import path
from keras.metrics import MeanAbsoluteError
from keras import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from scipy.spatial import distance

def create_data(IM_DIR, ethnie, proportion): 
    # Récupère les images d'un dossier et les associe à une étiquette - crée une structure avec noms - étiquettes - images
    prop = {0: [56, 56], 10: [50, 62], 20: [42, 70], 30: [32, 80], 40: [19, 93], 50: [0, 112]}
    os.chdir(IM_DIR)
    data = []
    for imname in os.listdir(IM_DIR):
        data.append([np.array(cv2.imread(imname, cv2.IMREAD_GRAYSCALE)), imname])
    np.random.shuffle(data)
    # Réduit le nombre de visages dans le jeu de données d'entraînement
    a_enlever, cpt_final = prop[proportion][0], prop[proportion][1]
    i = 0
    while a_enlever > 0:
        if ethnie in data[i][1]:
            a_enlever -= 1
            data.pop(i)
        else:
            i += 1
    return data, 56 - a_enlever, cpt_final

def create_model(nb_cc_value, nb_im):
    model = Sequential() # Ajoute les couches du modèle

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(300, 300, 1)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(units=nb_cc_value, activation='relu')) 

    model.add(Dense(nb_im, activation='sigmoid'))
    
    model.compile(optimizer=SGD(learning_rate=1, momentum=0.75), loss= 'mse', metrics=MeanAbsoluteError())
    
    return model

def save_loss(history, nb_cc_value, ethnie, proportion):
    d = history.history
    fig = plt.figure()
    plt.plot(range(epoques), d.get('loss'))
    plt.xlabel('Époques')
    plt.ylabel('Perte')
    fig.savefig(fname='C:/Users/Guilem/Documents/GitHub/Stage-L3-CNN-ORE/2eme_Semestre/ID_Results/'+str(nb_cc_value)+"_"+ethnie+"_"+str(proportion)+"_"+'.png')

def processing_data(data, nb_im):
    pairs_train = []
    name_train = []

    for i in range(0, len(data)):
            pairs_train.append(data[i][0]) # Image sous forme de matrice
            name_train.append(data[i][1])  # Nom de l'image
            
    xtrain = np.array(pairs_train).astype(int)
    ytrain = to_categorical(np.arange(nb_im)) # Encodage one-hot des valeurs cibles

    # Remodeler le jeu de données pour un seul canal
    width = xtrain.shape[1] # Dimension de l'image
    xtrain = xtrain.reshape((nb_im, width, width, 1))

    # Créer un générateur (1.0/255.0 = 0.003921568627451)
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    return xtrain, ytrain, name_train, datagen.flow(xtrain, ytrain, batch_size=64)

# %%--------------------------------------------------Initialisation
dirname = path.dirname('C:/Users/Guilem/Documents/GitHub/Stage-L3-CNN-ORE/2eme_Semestre')
IM_DIR = path.dirname('C:/Users/Guilem/Documents/GitHub/Stage-L3-CNN-ORE/STIM_NB_LumNorm')
results_csv_path = path.join(dirname, 'results.csv')
if not path.exists(results_csv_path):
    with open(results_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['run', 'nb_cc','ethnie','proportion', 'name_train', 'correct'])
# Construction de l'analyseur d'arguments et analyse des arguments
args = globals().get('args', None)
nb_cc_value = args.hidden
ethnie = args.ethnie
proportion = args.proportion
epoques = args.epoques
print(nb_cc_value, ethnie, proportion)

# Création des données à partir du dossier
data, nb_reduit, nb_im = create_data(IM_DIR, ethnie, proportion)
os.chdir(dirname)

# Traitement des données
xtrain, ytrain, name_train, train_iterator = processing_data(data, nb_im)

#%%----------------------------------------Entraînement 
model = create_model(nb_cc_value, nb_im)
history = model.fit(train_iterator, steps_per_epoch=len(train_iterator), epochs=epoques)

# Traçage et sauvegarde de la courbe d'entraînement
save_loss(history, nb_cc_value, ethnie, proportion)

#%%----------------------------------------Test 

prediction= model.predict(xtrain)
# Sans lab_train mais avec ytrain :
cpt_correct = 0
with open(results_csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(prediction)):
        correct_vector_distance = distance.euclidean(prediction[i], ytrain[i])
        other_distances = [distance.euclidean(prediction[i], ytrain[j]) for j in range(nb_im) if j != i]
        if correct_vector_distance <= min(other_distances):
            correct = 1
            cpt_correct += 1
        else:
            correct = 0
        writer.writerow([args["run"], args["hidden"], args["ethnie"], args["proportion"], name_train[i], correct])
print("Nombre de réponses correctes: ", cpt_correct, 'Précision: ', np.round((cpt_correct / len(prediction)), 3))
# Sauvegarde des résultats détaillés
xtrain = xtrain + 0.0
prediction = model.predict(xtrain)
#######################################################################################
# Sauvegarde des résultats MOYENS en dissociant CAU et ASIAN 
cpt_correct_ch, cpt_correct_cau = 0, 0
for i in range(len(prediction)):
    # Distance entre la prédiction et la vérité
    correct_vector_distance = distance.euclidean(prediction[i], ytrain[i])
    # Distances entre la pred et tous les autres vecteurs
    other_distances = [distance.euclidean(prediction[i], ytrain[j]) for j in range(nb_im) if j != i]
    #vérifie si la distance est bien la plus petite de toutes
    is_correct_prediction = correct_vector_distance <= min(other_distances)
    if "ch" in name_train[i]:
        cpt_correct_ch += is_correct_prediction
    elif "cau" in name_train[i]:
        cpt_correct_cau += is_correct_prediction

# proportions d'essais corrects, ajout d'un epsilon pour pas diviser par zéro
proportion_correct_ch = cpt_correct_ch / (len([name for name in name_train if "ch" in name]) + 2.220446049250313e-16)
proportion_correct_cau = cpt_correct_cau / (len([name for name in name_train if "cau" in name]) + 2.220446049250313e-16)
# f2.write("\n %s %s %s %s %f %f" % (args["run"], args["hidden"], args["ethnie"], args["proportion"], proportion_correct_cau, proportion_correct_ch))
# f2.close()

summary_csv_path = path.join(dirname, 'summary_results.csv')
if not path.exists(summary_csv_path):
    with open(summary_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # nom variables
        writer.writerow(['run', 'hidden', 'ethnie', 'proportion', 'proportion_correct_cau', 'proportion_correct_ch'])
with open(summary_csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([args["run"], args["hidden"], args["ethnie"], args["proportion"], proportion_correct_cau, proportion_correct_ch])

del data, xtrain, ytrain, prediction, model, history
gc.collect()