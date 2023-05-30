import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense,Reshape,MaxPooling2D,TimeDistributed, Dropout,LSTM, Conv3D,LayerNormalization
from keras.utils import to_categorical, Sequence
import numpy as np
import math
import cv2
import os
import glob
import random 
import datetime
random.seed(42)
np.random.seed(42) 

def gen(lista,batchSize=32,nRows=80,nCol=60):
    while True:
        random.shuffle(lista)
        count = 0
        results = []
        label = []
        for i in lista:
            result = []
            video = cv2.VideoCapture(i[0])
            while(video.isOpened()):
                ret, frame = video.read()
                if ret == True and count<100:
                        result.append(cv2.resize(frame, (nRows, nCol)))
                        count+=1
                else:
                    break
            try:
                x = random.sample(result, 30)
            except:
                x = random.sample(result, len(result))
            x = np.asarray(x)
            x = (x.astype(np.float32) / 255)-0.5
            try:
                x = np.reshape(x, (30,nRows,nCol))
                np.append(results,x)
                label.append(i[1])
            except:
                x = x + np.zeros_like((30,nRows,nCol)) 
                np.append(results,x)
                label.append(100)
            if count>=batchSize:
                count = 0 
                results = np.asarray(result)
                result = []
                yield [results, to_categorical(label)]




def load_groups(input_folder): 
    ''' 
    Loading the list of sub-folders into a python list with their 
    corresponding label. 
    '''
    listDir = os.listdir(input_folder)
    listDir.sort()
    groups = []
    index = 0
    for dir in listDir:
        videos =  glob.glob(f"{input_folder}/{dir}/*/*.mpg")
        for i in videos:
            groups.append([i,index]) 
        index+=1
    return groups


def get_file_list(train_groups): 
    train = [] 
    for video_dir in train_groups: 
        video_reader = cv2.VideoCapture(video_dir[0]) 
        if video_reader and video_reader.get(cv2.CAP_PROP_FRAME_COUNT) >= 31: 
            train.append([video_dir[0], video_dir[1]]) 
    return train

def split_data(groups): 
    ''' 
    Splitting the data at random for train and test set. 
    ''' 
    group_count = len(groups) 
    indices = np.arange(group_count) 
 
    np.random.shuffle(indices) 
 
    #60% training, 20% val and 20% test. 
    train_count = int(0.6 * group_count) 
    val_ind = int(0.8 * group_count) 
 
    train_groups = [groups[i] for i in indices[:train_count]] 
    val_groups  = [groups[i] for i in indices[train_count:val_ind]] 
    test_groups  = [groups[i] for i in indices[val_ind:]] 
 
    train = get_file_list(train_groups) 
    val = get_file_list(val_groups) 
    test = get_file_list(test_groups) 
 
    return train, val, test



#https://medium.com/smileinnovation/how-to-work-with-time-distributed-data-in-a-neural-network-b8b39aa4ce00
#Da rifare per mettere i dati distribuiti nel tempo



def createModel(inputshape):

    model = Sequential()

    # Aggiungi un layer TimeDistributed per applicare la stessa sequenza di operazioni a tutti i frame
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=inputshape))
    #model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu')))
    #input_shape=(32,30,60,80,3)

    model.add(TimeDistributed(Reshape((inputshape[1], inputshape[-3:-1], inputshape[-1]))))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))

    # Aggiungi altri layer TimeDistributed se necessario
    #model.add(TimeDistributed(Dense(64, activation='relu')))
    #model.add(TimeDistributed(Dense(64, activation='relu')))

    # Aggiungi un layer Dense finale per la classificazione
    model.add(Dense(11, activation='softmax'))


    model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=[keras.metrics.CategoricalAccuracy()]
    )

    print("modello creato")
    return model


callback = keras.callbacks.EarlyStopping(monitor= "val_loss", patience=2)
files = load_groups("UCF11_updated_mpg")
fileList = get_file_list(files)
train, val, test = split_data(fileList)
 

    
#Devo creare tre generatori con le liste differenti, in modo da fare un generatore per il train, 
# uno per il validation ed uno per il test




trainGen = gen(train)
valGen = gen(val)
testGen = gen(test)

input_shape=(32,30,60,80,3)


model = createModel(input_shape)
model.build()
model.summary()

model.fit(
    x = trainGen, epochs=1000,
    validation_data=valGen,
    callbacks=callback
)

results = model.evaluate(
  x = testGen
)

print(results)
now = datetime.datetime.now()

note=" "
with open('results3D.txt','a') as f:
  f.write(f"\n  {str(now.hour)}:{ str(now.minute)}  -  note: {note},  {results}")
