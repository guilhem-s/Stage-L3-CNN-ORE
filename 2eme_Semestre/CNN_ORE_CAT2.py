# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 22:57:02 2021
@author: LE 
Program for the training and testing of a CNN for a categorization task of faces as
asian or caucasian.
Convolutionnal layers pre-trained from a model with 50 neurons in the hidden layer, 
and an identification task.

Can be run with argparse parameters :
   a - hidden size 0 -> 8
   b - group to be reduced (ethnie) 0 1
   c - proportion of the reduction 0 -> 90
   d- number of run 
    
'
"""

import cv2, os
import numpy as np
from os import path
from keras import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import matplotlib.pyplot as plt


#%% Fonctions

def create_data(IM_DIR, ethnie, proportion): 
    #takes images from folder and associates them with a label 
    # create a structure with names - labels - im
    os.chdir(IM_DIR)
    prop = {0: [56, 56], 10: [50, 62], 20: [42, 70], 30: [32, 80], 40: [19, 93], 50: [0, 112]}
    data = []
    dataTEST = []
    cpt=0 #compteur pour les différentes images du dossier
    cpt_ch=0
    cpt_cau=0
    for imname in os.listdir(IM_DIR):
        if cpt== len(os.listdir(IM_DIR)): break
        if ("ch" in imname):
            label_im = [0,1]
        else:
            label_im = [1,0]
        img = cv2.imread(imname, cv2.IMREAD_GRAYSCALE)
        data.append([np.array(img), label_im, imname])  
        cpt += 1
    np.random.shuffle(data)
    
    #save 10 images for test set (5 caucasian and 5 chinese)
    i=0
    cpt_ch=0
    cpt_cau=0
    while len(dataTEST)<10:
        if ("ch" in data[i][2]) and (cpt_ch < 5):
            cpt_ch+=1
            dataTEST.append(data.pop(i))
        elif ("cau" in data[i][2]) and (cpt_cau < 5):
            cpt_cau+=1
            dataTEST.append(data.pop(i))
        i+=1

    prop = {0: [51, 51], 10: [46, 56], 20: [39, 63], 30: [30, 72], 40: [17, 85], 50: [0, 102]}
    a_enlever, cpt_final = prop[proportion][0], prop[proportion][1]   #56 - nb d'images à conserver
    i=0
    while a_enlever>0:
        if ethnie in data[i][2]:
            a_enlever-=1
            data.pop(i)
        else:
            i+=1
    
    return data, dataTEST, cpt_final

def create_model(nb_cc_value): #from pre-trained conv layers
    
    model = Sequential() #add model layers

    model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 1)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(nb_cc_value, activation='relu')) 

    model.add(Dense(1, activation='sigmoid'))
    
    return model

def save_loss(history, nb_cc_value, ethnie, proportion):
    d = history.history
    fig=plt.figure()
    plt.plot(range(epoques), d.get('loss'))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    fig.savefig('ID_Results/'+str(nb_cc_value)+"_"+ethnie+"_"+str(proportion)+"_"+'.png')

def processing_data(data, dataTEST, nb_im):
    pairs_train=[]
    lab_train = []
    name_train= []
    pairs_test= []
    lab_test = []
    name_test= []

    for i in range(len(dataTEST)):
        pairs_test.append(dataTEST[i][0])
        lab_test.append(dataTEST[i][1])
        name_test.append(dataTEST[i][2]) 

    for i in range(len(data)):
            pairs_train.append(data[i][0]) # image sous forme de matrice
            lab_train.append(data[i][1])
            name_train.append(data[i][2])  # nom de l'image
        
    xtrain=np.array(pairs_train).astype(int)
    ytrain = to_categorical(np.arange(nb_im)) # one hot encode target values

    xtest = np.array(pairs_test).astype(int)
    ytest = to_categorical(9)

    # reshape dataset to have a single channel
    width = xtrain.shape[1] # dimension de l'image
    print(xtrain.shape)
    xtrain = xtrain.reshape((nb_im, width, width, 1))
    xtest = xtest.reshape((10, width, width, 1))


    # create generator (1.0/255.0 = 0.003921568627451)
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    # prepare an iterators to scale images
    """# confirm scale of pixels
    print('Train min=%.3f, max=%.3f' % (xtrain.min(), xtrain.max()))
    print('Batches train=%d' % (len(train_iterator)))
    batchX, batchy = train_iterator.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max())) """
    
    return xtest, ytest, lab_test, name_test, datagen.flow(xtrain,ytrain, batch_size=64), datagen.flow(xtest, ytest, batch_size=64)

def save_loss(history, nb_cc_value, ethnie, proportion):
    d = history.history
    fig=plt.figure()
    plt.plot(range(epoques), d.get('loss'))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    fig.savefig('CAT_Results/'+str(nb_cc_value)+"_"+ethnie+"_"+str(proportion)+"_"+'.png')

# %%--------------------------------------------------Initialization
dirname= os.path.dirname("C:/Users/Guilem/Stage/CNN_ORE/")
IM_DIR= path.join(dirname, 'STIM_NB_LumNorm')

# construct the argument parser and parse the arguments
args = globals().get('args', None)
nb_cc_value = args.hidden
ethnie = args.ethnie
proportion = args.proportion
epoques = args.epoques

print(nb_cc_value, ethnie, proportion)

#create data from folder
data, dataTEST, nb_im = create_data(IM_DIR, ethnie, proportion)
outfile = "CAT_Results/"+str(nb_cc_value)+"_"+ethnie+"_"+str(proportion)+"_"+".txt"
os.chdir (dirname)

f2= open(path.join(dirname,'ALL_RES.txt'),"a+")
#Processing data
xtest, ytest, lab_test, name_test,train_iterator, test_iterator = processing_data(data, dataTEST, nb_im)
#%%----------------------------------------TRAIN  
opt = Adam()              
opt2 = SGD(learning_rate=0.1, momentum=0.9) # descente de gradient 
model = create_model(nb_cc_value)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_iterator, steps_per_epoch=len(train_iterator), epochs=epoques)

#plot and save training curve
save_loss(history, nb_cc_value, ethnie, proportion)

#%%----------------------------------------TEST 

results, acc = model.evaluate(test_iterator, steps=len(test_iterator), verbose=0)

#save detailed results
xtest=xtest+0.0
prediction= model.predict(xtest)

#save MEAN results dissociating CAU and ASIAN 

cpt_correct_ch= cpt_correct_cau=0
for i in range(len(prediction)): 
    if ("ch" in name_test[i]):
        if np.argmax(prediction[i]) == np.argmax(lab_test[i]) :
            cpt_correct_ch += 1
    if ("cau" in name_test[i]):
        if np.argmax(prediction[i]) == np.argmax(lab_test[i]) :
            cpt_correct_cau += 1

f2.write("\n %i, %s, %i, %f, %f" %(nb_cc_value, ethnie, proportion, cpt_correct_cau/5, cpt_correct_ch/5 ))
f2.close()