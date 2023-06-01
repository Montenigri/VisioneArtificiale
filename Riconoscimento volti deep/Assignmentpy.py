import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2d, Dropout,RandomBrightness,RandomRotation,RandomFlip
from keras.utils import to_categorical
from ultralytics import YOLO
import cv2
import numpy as np
import glob
import os
from math import floor

#https://github.com/derronqi/yolov8-face
#https://docs.ultralytics.com/
detect = YOLO('yolov8n-face.pt')

nomi = ["Davide","Francesco", "Gabriele", "Stefano", "Unknown"]


def getDataset(root="train"):

    listDir = os.listdir(root)
    tagFoto = []

    for dir in listDir:
        imgs =  glob.glob(f"{root}/{dir}/*.jpg")
        tagFoto.append([imgs,dir])

    for i in range(nomi):
        for k in tagFoto:
            if k[1] == nomi[i]:
                k[1] = i
    return tagFoto


def getSets(dataset, percentage=[0.6,0.2]):
    length = len(dataset)
    
    trainLen = floor(percentage[0]*length)
    valLen = floor(percentage[1]*length) + trainLen
    for i in dataset:
            i[0] = findFaces(i[0])
    train = dataset[:trainLen]
    val = dataset[trainLen:valLen] 
    test  = dataset [valLen:]

    return train, val, test


def findFaces(path):
    im1 = cv2.imread(path)
    img_test = detect.predict(source=im1)
    faces = []
    for result in img_test:
        boxes = result.boxes  
        boxes = boxes.numpy()
        face = im1[int(boxes.xyxy[0][1]):int(boxes.xyxy[0][3]),int(boxes.xyxy[0][0]):int(boxes.xyxy[0][2]),:]
        face = cv2.resize(face, (64,64))
        faces.append()
    return faces  

def splitXY(set):
    x=[]
    y=[]
    for i in set:
        x = i[0]
        y = i[1]
    return x,y


#Get dei dati

dataset = getDataset()
train, val, test = getSets(dataset)
X_train, Y_train = splitXY(train)
X_val,Y_val = splitXY(val)
X_test, Y_test = splitXY(test)


#Dobbiamo fare data augmentation qui

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip(),
  layers.RandomRotation(0.2),
  layers.RandomBrightness((-0.3,0.3))
])

#A questo punto dobbiamo riconoscere i volti con una rete neurale

model = Sequential()
model.add(data_augmentation)
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPool2d(pool_size=(2,2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPool2d(pool_size=(2,2)))
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



#qui dobbiamo ciclare ogni frame del video, per ogni frame ritagliare il volto
#passarlo alla rete neurale, farlo riconoscere e poi attaccare la label all'immagine