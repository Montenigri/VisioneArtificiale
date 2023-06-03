import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout,RandomBrightness,RandomRotation,RandomFlip,RandomZoom
from keras.utils import to_categorical
from ultralytics import YOLO
import cv2
import numpy as np
import glob
import os
from sklearn.utils import shuffle
from math import floor
from itertools import chain
from tqdm import tqdm
import pickle


#https://github.com/derronqi/yolov8-face
#https://docs.ultralytics.com/
detect = YOLO('yolov8n-face.pt')

nomi = ["Davide","Francesco", "Gabriele", "Stefano", "Unknown"]


def getDataset(root="train"):

    listDir = os.listdir(root)
    foto = []
    tag = []

    for dir in listDir:
        imgs =  glob.glob(f"{root}/{dir}/*.jpg")
        dirs = [dir]*len(imgs)
        foto = list(chain(foto,imgs))
        tag = list(chain(tag,dirs))
        


    for k in tqdm(range(len(tag)), desc= "Sostituisco nomi con interi"):
        for i in range(len(nomi)):
            if tag[k] == nomi[i]:
                tag[k] = i

    tag = list(map(int, tag))
    foto,tag = shuffle(foto,tag, random_state=42)
    return foto[:10],tag[:10]


def getSets(x,y, percentage=[0.6,0.2]):
    length = len(x)
    
    trainLen = floor(percentage[0]*length)
    valLen = floor(percentage[1]*length) + trainLen

    for i in tqdm(range(len(x)), desc= "Detecting faces"):
        x[i],_ = findFaces(cv2.imread(x[i]),maxDet=1)

    x = list(map(np.asarray, x))
    
    x = np.array(x)
    y = np.array(y)
    train = (x[:trainLen],y[:trainLen])
    val = (x[trainLen:valLen],y[trainLen:valLen]) 
    test  = (x[valLen:],y[valLen:])

    return train, val, test


def findFaces(frame, maxDet = 10):
    img_test = detect.predict(source=frame,max_det=maxDet,verbose=False)
    faces = []
    boxesDetect = []
    for result in img_test:
        boxes = result.boxes  
        boxes = boxes.numpy()
        face = frame[int(boxes.xyxy[0][1]):int(boxes.xyxy[0][3]),int(boxes.xyxy[0][0]):int(boxes.xyxy[0][2]),:]
        #Da mettere con padding per non storpiare le facce
        face =  cv2.resize(face, (64,64))
        faces = list(chain(faces,face))
        boxesDetect = list(chain(boxesDetect,boxes))
    
    return faces, boxesDetect



#Get dei dati

x,y = getDataset()

train, val, test = getSets(x,y)


with open("train.pkl","wb") as ds:
    pickle.dump(train,ds)

with open("val.pkl","wb") as ds:
    pickle.dump(val,ds)

with open("test.pkl","wb") as ds:
    pickle.dump(test,ds)


(X_train, Y_train) = train
(X_val,Y_val) = val
(X_test, Y_test) = test


#Dobbiamo fare data augmentation qui

data_augmentation = tf.keras.Sequential([
  RandomFlip("horizontal"),
  RandomRotation(0.2),
  RandomBrightness((-0.2,0.2)),
  RandomZoom(.1, .1)
])

Xtrain = []
Ytrain = []
for i in tqdm(range(len(X_train)), desc= "data augmentation"):
        Xtrain.append(data_augmentation(X_train[i]))
        Xtrain.append(X_train[i])
        Ytrain.append(Y_train[i])
        Ytrain.append(Y_train[i])

X_train = np.array(Xtrain)
Y_train = np.array(Ytrain)

#A questo punto dobbiamo riconoscere i volti con una rete neurale

model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))

model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=[keras.metrics.CategoricalAccuracy()]
)


model.summary()

callback = keras.callbacks.EarlyStopping(monitor= "val_loss", patience=3)



model.fit(
    X_train, to_categorical(Y_train), epochs=100, 
    batch_size=64, shuffle=True, 
    validation_data=(X_val,to_categorical(Y_val)),
    callbacks=callback
    )

#model.save_weights('modelClassificatore.h5')

results = model.evaluate(
  X_test,
  to_categorical(Y_test)
)
print(results)
#qui dobbiamo ciclare ogni frame del video, per ogni frame ritagliare il volto
#passarlo alla rete neurale, farlo riconoscere e poi attaccare la label all'immagine

def classificatore(frames):
    #Per mantenere l'univocità dei volti sui frame, ciclo singolamente i frame per poi inserire le
    #box ed i nomi
    font = cv2.FONT_HERSHEY_SIMPLEX

    for f in range(len(frames)):
        faces,boxes = findFaces(frames[f])
      
        faces = np.array(faces)
        faces = np.expand_dims(faces, axis=0)
        #https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
      
        predict = model.predict(faces)
        
        for (boxe,pred) in zip(boxes, predict[0]):
            frames[f] = cv2.putText(frames[f],nomi[int(pred)], (int(boxe.xyxy[0][1])-5,int(boxe.xyxy[0][3])-5),font, 1,(255,255,255),2)
            frames[f] = cv2.rectangle(frames[f], (int(boxe.xyxy[0][1]), int(boxe.xyxy[0][3])), (int(boxe.xyxy[0][0]), int(boxe.xyxy[0][2])), (255, 0, 255), 4)
        #Qua da vedere che viene restituito per poi attaccare i nomi alle facce e stamparle sul video
       
    return frames
    


#Raccolgo i frame e li passo al classificatore

video = cv2.VideoCapture("Video finale senza riconoscimento.mp4")
frames = []
if (video.isOpened() == False):
    print("Error opening video file")
while(video.isOpened()):
  ret, frame = video.read()
  if ret == True:
        frames.append(frame)
  else:
      break

results = classificatore(frames[:30])
print(np.shape(results[0]))
height, width, channels = results[0].shape
size = (width,height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out15 = cv2.VideoWriter('project_video_finale.avi',fourcc, 15, size)

for i in tqdm(range(len(results)), desc="Salvataggio frames"):
    out15.write(results[i])
out15.release()
