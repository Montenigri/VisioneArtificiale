import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense,BatchNormalization,MaxPooling2D,TimeDistributed, LTSM
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

class dataGenerator(Sequence):
    def __init__(self, lista,batchSize,nRows=80,nCol=60,shuffle):
        self.lista = lista
        self.batchSize = batchSize
        self.nRows = nRows
        self.nCol = nCol
        self.list_IDs = np.arange(len(lista))
        self.shuffle = shuffle


    def __len__(self):
        return int(np.floor(len(self.lista)/self.batchSize))
    
    def on_epoch_end(self): 
        self.indexes = self.list_IDs  #Load the indexes of the data
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        #Generate batch at position 'index' 
        index = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #Generate a temporary list of indexes that forms a batch based on  ##the index selected above.
        list_IDs_temp = [self.list_IDs[k] for k in index]
        #Generate batch
        X,y = self.__data_generation(list_IDs_temp)
        return X,y
    
    def __data_generation(self,list_IDs_temp):
        X_data = []
        y_data = []
        for i,_ in enumerate(list_IDs_temp):
            batchSamples = self.data.iloc[i,0]
        y = self.data.iloc[i,1]
        temp_data_list = []
        #Il problema è qui, con le immagini è facile, con i video meno
        #Preprocessing che mi divide tutti i video in frame scalati (?)
        #Però deve caricare tutto il video per forza

        #Faccio diventare ogni video una serie di immagini, passo la serie di immagini con la label,
        #La rete deve avere il blocco per gestire la temporabilità delle immagini
        for img in batchSamples:
            try:
                image = cv2.imread(img,0)
                ext_img = cv2.resize(image,(self.nRows,self.nCol))
            except Exception as e:
                print('Value error ',e)
        temp_data_list.append(ext_img)
        X_data.append(temp_data_list)
        y_data.append(y)
        X = np.array(X_data)
        y = np.array(y_data)
        
        return X, to_categorical(y)

#Devo modificare il generatore affinché presa una lista faccia la generazione solo su quella, datagen deve fare quello che faceva il generatore fatto a manina
#Devo caricare i frame invece delle foto singole, quindi a mano devo caricare n frame e poi andare avanti
#Esempio https://gist.github.com/metal3d/f671c921440deb93e6e040d30dd8719b#:~:text=def%20__openframes(self%2C%20classname%2C%20file)%3A
#https://www.tensorflow.org/tutorials/load_data/video

#Il secondo giro del generatore torna 0 elementi e rompe tutto, questo è possibilmente dato dall'apertura del video,
#trovare un modo per chiuderlo e poi riaprire il prossimo, magari svuotando 

def gen(lista,batchSize,nRows=80,nCol=60):
    random.shuffle(lista)
    count = 0
    while True:
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
            print(len(result))
            x = random.sample(result, 30)
            x = np.asarray(x)
            x = (x.astype(np.float32) / 255)-0.5
            x = np.reshape(x, (30,nRows,nCol,3))
            results.append(x)
            label.append(i[1])
            if count==batchSize:
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
        if video_reader and video_reader.get(cv2.CAP_PROP_FRAME_COUNT) >= 16: 
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

    model.add(
        TimeDistributed(
            Conv2D(16, (3,3), activation='relu', input_shape=inputshape)
            )
        )
    model.add(
        TimeDistributed(
            BatchNormalization(axis=1, synchronized = True )
            )
        )
    model.add(
        TimeDistributed(
            Conv2D(8, (3,3), activation='relu')
            )
        )
    model.add(
        TimeDistributed(
            MaxPooling2D(pool_size=(5,5))
            )
        )
    #Controllare dimensione della ltsm
    #model.add(LTSM(1024, activation='relu', return_sequences=False))
    model.add(Flatten())
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
trainGen = dataGenerator()
valGen = dataGenerator()
testGen = dataGenerator()

model = createModel((32,80,60,3))
model.build()
model.summary()
model.fit(
    x = gen(train,32), epochs=1000,
    validation_data=gen(val,32),
    callbacks=callback
)

results = model.evaluate(
  x = gen(test,32)
)

print(results)
now = datetime.datetime.now()

note=" "
with open('results3D.txt','a') as f:
  f.write(f"\n  {str(now.hour)}:{ str(now.minute)}  -  note: {note},  {results}")
