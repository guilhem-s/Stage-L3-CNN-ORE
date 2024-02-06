import cv2, os
import tensorflow
import keras
import numpy as np
from os import fspath, path
from keras import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import argparse

config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.compat.v1.Session(config=config)

# Utiliser la session TensorFlow
tensorflow.compat.v1.keras.backend.set_session(session)


def create_data(IM_DIR, ethnie, proportion): 
    #takes images from folder and associates them with a label - create a structure with
    #names - labels - im
    os.chdir(IM_DIR)
    data = []
    dataTEST = []
    label = []
    cpt=0  #compte le nombre d'images à traiter
    to_remove = []
    for imname in os.listdir(IM_DIR): # pour chaque image dans le répertoire
        if cpt== len(os.listdir(IM_DIR)): 
            break # on sort de la boucle quand on a traité toutes les images

        name_label = imname
        print(name_label)
        label_im = [name_label] 
        label.append(name_label)

        img = cv2.imread(imname, cv2.IMREAD_GRAYSCALE) #convertie l'image en matrice
        data.append([np.array(img), label_im, imname])  
        cpt += 1
    np.random.shuffle(data)
            
    # reduce the number of faces from one group  in the train set
    ethnie_reduced = ["ch","cau"][ethnie]
    reduced_db = cpt - int( (cpt//2) * (proportion/100) ) #nb d'images à supprimer
    compteur = 0

    for i in range(0,len(data)):
        if (ethnie_reduced in data[i][2]) & (compteur<cpt-reduced_db):
            to_remove.append(i)
            compteur += 1
    for ele in sorted(to_remove, reverse=True):
        del data[ele]
    
    for i in range(0,len(data)):
        data[i][1] = i

    for i in range(0,len(dataTEST)):
        dataTEST[i][1] = i

    np.random.shuffle(data)
    return data, dataTEST

def rmse_loss(y_true, y_pred):
    return tensorflow.keras.backend.sqrt(tensorflow.keras.losses.MSE(y_true, y_pred))

def create_model(nb_cc, myconv, nb_im): #from pre-trained conv layers
    
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
data, dataTEST = create_data(IM_DIR, ethnie, proportion)
outfile = "CAT_Results_ID/"+str(args["hidden"])+"_"+str(args["ethnie"])+"_"+str(args["proportion"])+"_"+str(args["run"])+".txt"
os.chdir (dirname) 
#create result file
f1= open(path.join(dirname,outfile),"a+") #ouvre en modification dirname
f1.write("\nNB_CC IM ACCURACY\n" )

f2= open(path.join(dirname,'ALL_RES.txt'),"a+")

#Processing data
pairs_train=[]
pairs_test=[]
lab_train= []
lab_test=[]
name_train=[]
name_test=[]

for i in range(0,len(data)):
        pairs_train.append(data[i][0])
        lab_train.append(data[i][1])
        name_train.append(data[i][2])
for i in range(0,len(dataTEST)):
        pairs_test.append(dataTEST[i][0])
        lab_test.append(dataTEST[i][1])
        name_test.append(dataTEST[i][2])        
        
xtrain=np.array(pairs_train).astype(int)
ytrain=np.array(lab_train).astype(int)
xtrain = xtrain.reshape(-1, 300, 300, 1)

xtest=np.array(pairs_test)
ytest=np.array(lab_test)
xtest = xtest.reshape(-1, 300, 300, 1)
#%%----------------------------------------TRAIN 
opt = SGD(learning_rate=0.01, momentum=0.9) # descente de gradient

myconv = tensorflow.keras.models.load_model(path.join(dirname, 'saved_models'), custom_objects={'rmse_loss': rmse_loss}) # , compile=False
myconv.compile(loss=rmse_loss, optimizer = opt, metrics = ["accuracy"])

model = create_model(nb_cc_value, myconv, len(data))
model.compile(loss= 'sparse_categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])
history = model.fit(xtrain,ytrain, epochs=10,batch_size=10)

#plot and save training curve
d= history.history
fig=plt.figure()
plt.plot(range(10), d.get('loss'))
plt.xlabel('Epochs')
plt.ylabel('Loss')
fig.savefig('CAT_Results/'+str(args["hidden"])+"_"+str(args["ethnie"])+"_"+str(args["proportion"])+"_"+str(args["run"])+'.png')

#%%----------------------------------------TEST 

results=model.evaluate(xtest, ytest)


#save detailed results
xtest=xtest+0.0
prediction= model.predict(xtest)
cpt_correct=0
for i in range(0,len(prediction)): 
    if np.argmax(prediction[i]) == np.argmax(lab_test[i]) :
        correct=1
        cpt_correct=cpt_correct+1
    else:
        correct=0
    f1.write("\n %F %s %f " %(nb_cc_value, name_test[i], correct))
print(cpt_correct/len(prediction))

#save MEAN results dissociating CAU and ASIAN 

cpt_correct_ch= cpt_correct_cau=0
for i in range(0,len(prediction)): 
    if ("ch" in name_test[i]):
        if np.argmax(prediction[i]) == np.argmax(lab_test[i]) :
            correct_ch=1
            cpt_correct_ch=cpt_correct_ch+1
        else:
            correct_ch=0
    if ("cau" in name_test[i]):
        if np.argmax(prediction[i]) == np.argmax(lab_test[i]) :
            correct_cau=1
            cpt_correct_cau=cpt_correct_cau+1
        else:
            correct_cau=0    

f2.write("\n %s %s %s %s %f %f" %(args["run"], args["hidden"], args["ethnie"], args["proportion"], cpt_correct_cau/5, cpt_correct_ch/5 ))

f1.close()
f2.close() 