import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2d
from keras.utils import to_categorical
from ultralytics import YOLO
import cv2
import numpy as np
import glob
import os
import albumentations as A

#https://github.com/derronqi/yolov8-face
#https://docs.ultralytics.com/
detect = YOLO('yolov8n-face.pt')


#Questa cosa funziona solo per il training, in quanto poi bisonga prevedere l'arrivo di più 
#volti per immagine, per allenare è apposto cosi
root = ["train","val","test"]
sets = {}
for fol in root:
    listDir = os.listdir(fol)
    tagFoto = []

    for dir in listDir:
        imgs =  glob.glob(f"{root}/{dir}/*.jpg")
        tagFoto.append([imgs,dir])

    sets[fol]=tagFoto


#Questa cosa deve tornare per ogni immagine il volto e lo salviamo da qualche parte, 
#si può anche pensare di fare tre volte il ciclo, uno per set, cosi da dividere la quantità di memoria usata
# e si salva con un pkl o altro
im1 = cv2.imread("test/Davide_Sgroi2.jpg")
img_test = detect.predict(source=im1)
for result in img_test:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    probs = result.probs  # Class probabilities for classification outputs

boxes = boxes.numpy()
#print(boxes.xyxy[0][2])
im1Copy = im1.copy()
im1Copy = im1Copy[int(boxes.xyxy[0][1]):int(boxes.xyxy[0][3]),int(boxes.xyxy[0][0]):int(boxes.xyxy[0][2]),:]
cv2.imwrite("prova.jpg", im1Copy)


#A questo punto dobbiamo riconoscere i volti con una rete neurale

model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPool2d(pool_size=(2,2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPool2d(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

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