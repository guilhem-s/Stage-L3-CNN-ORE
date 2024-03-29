import cv2, os, gc, csv, argparse
import tensorflow
import numpy as np
from os import path
from keras import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
from scipy.spatial import distance

def create_data(IM_DIR, ethnie, proportion): 
    #takes images from folder and associates them with a label - create a structure with
    #names - labels - im
    os.chdir(IM_DIR)
    data = []
    to_remove = []
    for imname in os.listdir(IM_DIR):
        data.append([np.array(cv2.imread(imname, cv2.IMREAD_GRAYSCALE)), imname])

    # reduce the number of faces from one group  in the train set
    ethnie_reduced = ["ch","cau"][ethnie]
    reduced_db = 56 - int(56*proportion / (100-proportion)) #nb d'images à supprimer
    cpt_final = 112 - reduced_db
    
    for i in range(0,len(data)):
        if (ethnie_reduced in data[i][1]) & (reduced_db>0):
            to_remove.append(i)
            reduced_db-=1
            
    for ele in sorted(to_remove, reverse=True):
        del data[ele]
        
    for i in range(len(data)):
        random_vector = np.random.randint(0, 2, 56)  # vecteur de 56 valeurse entières : 0 ou 1
        data[i].insert(1, random_vector)

    np.random.shuffle(data)
    return data

def extract_components(data):
    images = [item[0] for item in data]
    vectors = [item[1] for item in data]
    names = [item[2] for item in data]
    images_array = np.array(images)
    vectors_array = np.array(vectors)
    return images_array, vectors_array, names

def create_dict_from_data(data):
    dict_data = {}
    for item in data:
        image_vector = item[1]  # vecteur
        image_name = item[2]    # nom de l'image
        dict_data[image_name] = image_vector
    return dict_data
# %%--------------------------------------------------Initialization
dirname= path.dirname(r'C:/Users/Guilem/Documents/GitHub/Stage-L3-CNN-ORE/')
IM_DIR= path.join(dirname, 'STIM_NB_LumNorm')
results_csv_path = path.join(dirname, 'results.csv')
if not path.exists(results_csv_path):
    with open(results_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['run', 'nb_cc','ethnie','proportion', 'name_train', 'correct'])

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--hidden", type=int,
                help="hidden size of the network")
ap.add_argument("-b", "--ethnie", type=str,
                help="ethnie reduced")
ap.add_argument("-c", "--proportion", type=int,
                help="reduction proportion")
ap.add_argument("-d", "--run", type=int,
                help="run id")
args = vars(ap.parse_args())

neurones = args.hidden

nb_cc_value = int(args["hidden"])
ethnie = int(args["ethnie"])
proportion = int(args["proportion"])
run = int(args["run"])

data = create_data(IM_DIR, ethnie, proportion)
dict_data = create_dict_from_data(data)
print('nb_cc: ',nb_cc_value, 'reduction :', proportion, 'ethnie : ', ethnie)
images_array, vectors_array, names = extract_components(data)

#create data from folder
data = create_data(IM_DIR, ethnie, proportion)
outfile = "2eme_Semestre/ID_Results/"+str(args["hidden"])+"_"+str(args["ethnie"])+"_"+str(args["proportion"])+"_"+str(args["run"])+".txt"
os.chdir(dirname)
f2= open(path.join(dirname,'ALL_RES.txt'),"a+")

#Processing data
pairs_train=[]
lab_train= []
name_train=[]

for i in range(0,len(data)):
        pairs_train.append(data[i][0]) # image sous forme de matrice
        lab_train.append(data[i][1])   # vecteur associé à l'image
        name_train.append(data[i][2])  # nom de l'image
        
xtrain=np.array(pairs_train).astype(float)
ytrain=np.array(lab_train).astype(float)
xtrain = xtrain.reshape(-1, 300, 300, 1)

xtest=np.array(pairs_train).astype(float)
ytest=np.array(lab_train).astype(float)
xtest = xtest.reshape(-1, 300, 300, 1)
#%%----------------------------------------TRAIN
lrelu = tensorflow.keras.layers.LeakyReLU(alpha=0.2)
model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Flatten())
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(tensorflow.keras.layers.Dense(900, activation = lrelu,
kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=42)))
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(tensorflow.keras.layers.Dense(900, activation = lrelu,
kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=42)))
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(tensorflow.keras.layers.Dense(a, activation = lrelu,
kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=42)))
model.add(tensorflow.keras.layers.BatchNormalization())
model.add(tensorflow.keras.layers.Dense(56, activation = 'sigmoid',
kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=42)))

lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.7, staircase=False)
optimizer = tensorflow.keras.optimizers.SGD(learning_rate=lr_schedule, momentum = 0.9)
model.compile(loss= 'mse', optimizer = optimizer, metrics = tensorflow.keras.metrics.MeanAbsoluteError())
history = model.fit(xtrain,ytrain, epochs=150,batch_size=32, shuffle=False, verbose=False)

#plot and save training curve
history_model = history.history
fig=plt.figure()
plt.plot(range(150), history_model.get('loss'))
plt.xlabel('Epochs')
plt.ylabel('Loss')
fig.savefig('ID_Results/'+str(args["hidden"])+"_"+str(args["ethnie"])+"_"+str(args["proportion"])+"_"+str(args["run"])+'.png')
plt.close()

#%%----------------------------------------TEST
prediction =  model(xtrain).numpy()
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
        writer.writerow([args["run"], args["hidden"], args["ethnie"], args["proportion"], name_train[i], correct])
print("nb correct: ", cpt_correct, 'accuracy: ', np.round((cpt_correct / len(prediction)), 3))
#######################################################################################
#save MEAN results dissociating CAU and ASIAN 
cpt_correct_ch, cpt_correct_cau = 0, 0
for i in range(len(prediction)):
    # distance pred à ground truth
    correct_vector_distance = distance.euclidean(prediction[i], lab_train[i])
    # distances entre la pred et tous les autres vecteurs
    other_distances = [distance.euclidean(prediction[i], lab_train[j]) for j in range(len(lab_train)) if j != i]
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

del data, xtrain, ytrain, xtest, ytest, prediction, model, history, fig
gc.collect()