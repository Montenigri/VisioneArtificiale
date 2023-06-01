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

    for k in range(len(tag)):
        for i in range(len(nomi)):
            if tag[k] == nomi[i]:
                tag[k] = i

    tag = list(map(int, tag))
    shuffle(foto,tag)
    return foto,tag


def getSets(x,y, percentage=[0.6,0.2]):
    length = len(x)
    
    trainLen = floor(percentage[0]*length)
    valLen = floor(percentage[1]*length) + trainLen

    for i in range(len(x)):
        x[i],_ = findFaces(cv2.imread(x[i]))

    train = (x[:trainLen],y[:trainLen])
    val = (x[trainLen:valLen],y[trainLen:valLen]) 
    test  = (x[valLen:],y[valLen:])

    return train, val, test


def findFaces(frame):
    img_test = detect.predict(source=frame,max_det=10,verbose=False)
    faces = []
    boxesDetect = []
    for result in img_test:
        boxes = result.boxes  
        boxes = boxes.numpy()
        face = frame[int(boxes.xyxy[0][1]):int(boxes.xyxy[0][3]),int(boxes.xyxy[0][0]):int(boxes.xyxy[0][2]),:]
        #Da mettere con padding per non storpiare le facce
        face = cv2.resize(face, (64,64))
        faces = list(chain(faces,face))
        boxesDetect = list(chain(boxesDetect,boxes))
    return faces, boxesDetect



#Get dei dati

x,y = getDataset()
train, val, test = getSets(x,y)
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



#qui dobbiamo ciclare ogni frame del video, per ogni frame ritagliare il volto
#passarlo alla rete neurale, farlo riconoscere e poi attaccare la label all'immagine

def classificatore(frames):
    #Per mantenere l'univocità dei volti sui frame, ciclo singolamente i frame per poi inserire le
    #box ed i nomi
    font = cv2.FONT_HERSHEY_SIMPLEX

    for f in range(len(frames)):
        faces,boxes = findFaces(frames[f])
        #https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
        predict = model.predict(faces)
        for (boxe,pred) in zip(boxes, predict):
            frames[f] = cv2.putText(frames[f],nomi[int(pred)], (boxe.xyxy[0][1]-5,boxe.xyxy[0][3]-5),font, 1,(255,255,255),2)
            frames[f] = cv2.rectangle(frames[f], (boxe.xyxy[0][1], boxe.xyxy[0][3]), (boxe.xyxy[0][0], boxe.xyxy[0][2]), (255, 0, 255), 4)
        #Qua da vedere che viene restituito per poi attaccare i nomi alle facce e stamparle sul video
    return frames
    

#fare tutto in una sola volta

#Raccolgo i frame e li passo al classificatore
video = cv2.VideoCapture("Video finale senza riconoscimento.mp4")
frames = []
if (video.isOpened()== False):
    print("Error opening video file")
while(video.isOpened()):
  ret, frame = video.read()
  if ret == True:
        frames.append(frame)
  else:
      break

results = classificatore(frames)

height, width = results[0].shape
size = (width,height)

fourcc = -1 

out15 = cv2.VideoWriter('project_video_finale.mp4',fourcc, 15, size)

for i in results:
    out15.write(i)
out15.release()




#Errore Make sure all arrays contain the same number of samples.
#Possibilmente ci sono falsi positivi che portano ad avere un numero di facce differente da quello aspettato