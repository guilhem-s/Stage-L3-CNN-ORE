# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:19:34 2023

@author: Guilem
"""

import cv2, os, gc, csv, argparse
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

def create_model(nb_cc_value, nb_im):
    
    model = Sequential() #add model layers

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(300, 300, 1)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(units=nb_cc_value, activation='relu')) 

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
            
    xtrain=np.array(pairs_train)
    ytrain = to_categorical(np.arange(nb_im)) # one hot encode target values

    # reshape dataset to have a single channel
    width = xtrain.shape[1] # dimension de l'image
    xtrain = xtrain.reshape((nb_im, width, width, 1))

    # create generator (1.0/255.0 = 0.003921568627451)
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    return xtrain, ytrain, name_train, datagen.flow(xtrain, ytrain, batch_size=64)

# %%--------------------------------------------------Initialization
directory= path.dirname('C:/Users/Guilem/Documents/GitHub/Stage-L3-CNN-ORE/')
IM_DIR= path.join(directory, 'STIM_NB_LumNorm')
results_csv_path = path.join(directory, 'results.csv')
if not path.exists(results_csv_path):
    with open(results_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['run', 'nb_cc','ethnie','proportion', 'name_train', 'correct'])
dirname = path.join(directory, '2eme_Semestre')

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
opt  = SGD(learning_rate=1, momentum=0.75) # descente de gradient 
model = create_model(nb_cc_value, nb_im)
model.compile(optimizer=opt, loss= 'mse', metrics=MeanAbsoluteError())
history = model.fit(train_iterator, steps_per_epoch=len(train_iterator), epochs=epoques)

#plot and save training curve
save_loss(history, nb_cc_value, ethnie, proportion)

#%%----------------------------------------TEST 

prediction= model.predict(xtrain)
#sans lab_train mais avec ytrain:
cpt_correct = 0
with open(results_csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(prediction)):
        correct_vector_distance = distance.euclidean(prediction[i], ytrain[i])
        other_distances = [distance.euclidean(prediction[i], ytrain[j]) for j in range(len(ytrain)) if j != i]
        if correct_vector_distance <= min(other_distances):
            correct = 1
            cpt_correct += 1
        else:
            correct = 0
        writer.writerow([nb_cc_value, ethnie, proportion, name_train[i], correct])
print("nb correct: ", cpt_correct, 'accuracy: ', np.round((cpt_correct / len(prediction)), 3))
#save detailed results
xtrain=xtrain+0.0
prediction= model.predict(xtrain)
#######################################################################################
#save MEAN results dissociating CAU and ASIAN 
cpt_correct_ch, cpt_correct_cau = 0, 0
for i in range(len(prediction)):
    # distance pred à ground truth
    correct_vector_distance = distance.euclidean(prediction[i], ytrain[i]) #lab_
    # distances entre la pred et tous les autres vecteurs
    other_distances = [distance.euclidean(prediction[i], ytrain[j]) for j in range(nb_im) if j != i] #lab_
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
        writer.writerow(['hidden', 'ethnie', 'proportion', 'proportion_correct_cau', 'proportion_correct_ch'])
with open(summary_csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([nb_cc_value, ethnie, proportion, proportion_correct_cau, proportion_correct_ch])

del data, xtrain, ytrain, prediction, model, history
gc.collect()
