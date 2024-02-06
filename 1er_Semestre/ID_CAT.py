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
from keras.optimizers import SGD
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import argparse

config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.compat.v1.Session(config=config)

# Utiliser la session TensorFlow
tensorflow.compat.v1.keras.backend.set_session(session)


def create_data(IM_DIR, ethnie, proportion): 
    # takes images from folder and associates them with a label - 
    # create a structure with names - labels - im
    os.chdir(IM_DIR)
    data = []
    to_remove = []
    for imname in os.listdir(IM_DIR):
        data.append([np.array(cv2.imread(imname, cv2.IMREAD_GRAYSCALE)),
                     imname])
            
    # Reduit le nombre de visages dans le training set
    ethnie_reduced = ["ch","cau"][ethnie]
    nb_reduit = int(56*proportion / (100-proportion)) #nb d'images à conserver
    reduced_db = 56 - nb_reduit #nb d'images à supprimer
    cpt_final = 112 - reduced_db # nombre d'images total dans le training set
    for i in range(0,len(data)):
        if (ethnie_reduced in data[i][1]) & (reduced_db>0):
            to_remove.append(i)
            reduced_db-=1
    for ele in sorted(to_remove, reverse=True):
        del data[ele]
        
    cpt=0 #compteur pour les différentes images du dossier
    while cpt<cpt_final:
        label_im = np.zeros(cpt_final, dtype = int) 
        label_im[cpt] = 1 # label image vecteur encodé one hot 
        data[cpt].insert(1, label_im) 
        cpt += 1
    np.random.shuffle(data)
    return data, nb_reduit

def rmse_loss(y_true, y_pred):
    return tensorflow.keras.backend.sqrt(tensorflow.keras.losses.MSE(y_true, y_pred))

def create_model(nb_cc_value, myconv, nb_im): #from pre-trained conv layers
    
    model = Sequential()#add model layers
    model.add(myconv.get_layer("conv2d_14"))
    model.add(myconv.get_layer("max_pooling2d_14"))
    model.add(myconv.get_layer("conv2d_15"))
    model.add(myconv.get_layer("max_pooling2d_15"))
    model.add(Flatten())
    model.add(Dense(nb_cc_value, activation='relu')) 
    model.add(Dense(nb_im, activation='softmax'))
    return model

# %%--------------------------------------------------Initialization
dirname= path.dirname('C:/Users/Guilem/Stage/CNN_ORE/')
IM_DIR= path.join(dirname, 'STIM_NB_LumNorm')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--hidden", type=int,
                help="hidden size of the network")
ap.add_argument("-b", "--ethnie", type=str,
                help="ethnie reduced")
ap.add_argument("-c", "--proportion", type=str,
                help="reduction proportion")
ap.add_argument("-d", "--run", type=str,
                help="run id")
args = vars(ap.parse_args())

list_cc=[2, 8, 14, 20, 26, 32, 38, 44, 50]
nb_cc_value = list_cc[int(args["hidden"])]
ethnie = int(args["ethnie"])
proportion = int(args["proportion"])
run = int(args["run"])
#create data from folder
data, nb_reduit = create_data(IM_DIR, ethnie, proportion)
outfile = "ID_Results/"+str(args["hidden"])+"_"+str(args["ethnie"])+"_"+str(args["proportion"])+"_"+str(args["run"])+".txt"
os.chdir (dirname) 
#create result file
f1 = open(path.join("C:/Users/Guilem/Stage/CNN_ORE/",outfile), "a+") # ouvre en modification dirname
f1.write("\nNB_CC IM ACCURACY\n" )

f2= open(path.join(dirname,'ALL_RES.txt'),"a+")

#Processing data
pairs_train=[]
label_train= []
name_train=[]

for i in range(0,len(data)):
        pairs_train.append(data[i][0]) # image sous forme de matrice
        label_train.append(data[i][1])   # vecteur associé à l'image
        name_train.append(data[i][2])  # nom de l'image
        
xtrain=np.array(pairs_train).astype(int)
ytrain=np.array(label_train).astype(int)
xtrain = xtrain.reshape(-1, 300, 300, 1)
#%%----------------------------------------TRAIN 
opt = SGD(learning_rate=0.001, momentum=0.9) # descente de gradient 

myconv = tensorflow.keras.models.load_model(path.join(dirname, 'saved_models'), custom_objects={'rmse_loss':rmse_loss})
myconv.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

model = create_model(nb_cc_value, myconv, len(data))
model.compile(loss= 'categorical_crossentropy',
              optimizer = 'Adam', metrics = ["accuracy"])
history = model.fit(xtrain,ytrain, epochs=40,batch_size=32)

#plot and save training curve
d = history.history
fig=plt.figure()
plt.plot(range(40), d.get('loss'))
plt.xlabel('Epochs')
plt.ylabel('Loss')
fig.savefig('ID_Results/'+str(args["hidden"])+"_"+str(args["ethnie"])+"_"+str(args["proportion"])+"_"+str(args["run"])+'.png')

#%%----------------------------------------TEST 

results=model.evaluate(xtrain, ytrain)


#save detailed results
xtrain=xtrain+0.0
prediction= model.predict(xtrain)
cpt_correct=0
for i in range(0,len(prediction)): 
    if np.argmax(prediction[i]) == np.argmax(label_train[i]) :
        correct=1
        cpt_correct=+1
    else:
        correct=0
    f1.write("\n %F %s %f " %(nb_cc_value, name_train[i], correct))
# print(cpt_correct/len(prediction))

#save MEAN results dissociating CAU and ASIAN 

cpt_correct_ch= cpt_correct_cau=0
for i in range(0,len(prediction)): 
    if ("ch" in name_train[i]):
        if np.argmax(prediction[i]) == np.argmax(label_train[i]) :
            cpt_correct_ch += 1
            
    if ("cau" in name_train[i]):
        if np.argmax(prediction[i]) == np.argmax(label_train[i]) :
            cpt_correct_cau += 1
            
cpt_ch = 0
cpt_cau = 0
for i in range(0,len(data)):
    if "ch" in data[i][2]:
        cpt_ch+=1
    elif "cau" in data[i][2]:
        cpt_cau+=1
s = "Il y a " + str(cpt_cau) + " caucasiens et " + str(cpt_ch) + " chinois"

if ethnie == 0:
    if cpt_correct_ch != 0: # On réduit les caucasiens, 56 visages chinois et le reste en caucasien
        f2.write("%i, %s, %s, %f, %f, %s \n" %(nb_cc_value, args["ethnie"], args["proportion"], cpt_correct_cau/56, cpt_correct_ch / nb_reduit, s ))
    else:
        f2.write("%i, %s, %s, %f, %f, %s \n" %(nb_cc_value, args["ethnie"], args["proportion"], cpt_correct_cau/56, 0, s ))

else:
    if cpt_correct_cau != 0:
        f2.write("%i, %s, %s, %f, %f, %s \n" %(nb_cc_value, args["ethnie"], args["proportion"], cpt_correct_cau / nb_reduit, cpt_correct_ch/56, s ))
    else:
        f2.write("%i, %s, %s, %f, %f, %s \n" %(nb_cc_value, args["ethnie"], args["proportion"], 0, cpt_correct_ch/56, s ))

f1.close()
f2.close()