import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout,RandomBrightness,RandomRotation,RandomFlip,RandomZoom,Resizing
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


resizer = Resizing(
    64,
    64,
    interpolation='bilinear',
    crop_to_aspect_ratio=True,
)

def getDataset(root="train"):

    listDir = os.listdir(root)
    foto = []
    tag = []

    for dir in listDir:
        imgs =  glob.glob(f"{root}/{dir}/*.jpg")
        dirs = [dir]*len(imgs)
        foto = list(chain(foto,imgs))
        tag = list(chain(tag,dirs))
        
    for k in tqdm(range(len(tag)), desc= "Changing names to index"):
        for i in range(len(nomi)):
            if tag[k] == nomi[i]:
                tag[k] = i

    tag = list(map(int, tag))
    foto,tag = shuffle(foto,tag, random_state=42)
    return foto,tag


def getSets(x,y, percentage=[0.6,0.2]):
    length = len(x)
    
    trainLen = floor(percentage[0]*length)
    valLen = floor(percentage[1]*length) + trainLen

    for i in tqdm(range(len(x)), desc= "Detecting faces"):
        x[i],_ = findFaces(cv2.imread(x[i]),maxDet=1)

    x = list(map(np.array, x))
    
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
        for b in boxes:
            face = frame[int(b.xyxy[0][1]):int(b.xyxy[0][3]),int(b.xyxy[0][0]):int(b.xyxy[0][2]),:]
            face = resizer(face)
            faces.append(face)
        boxesDetect = list(chain(boxesDetect,boxes))
    faces = np.array(faces, dtype=np.float32)
    return faces, boxesDetect



data_augmentation = tf.keras.Sequential([
  RandomFlip("horizontal"),
  RandomRotation(0.2),
  RandomBrightness((-0.2,0.2)),
  RandomZoom(.1, .1)
])


#Get dei dati

if os.path.exists("train.pkl") and os.path.exists("val.pkl") and os.path.exists("test.pkl"):
    with open("train.pkl","rb") as ds:
        train = pickle.load(ds)
    with open("val.pkl","rb") as ds:
        val = pickle.load(ds)
    with open("test.pkl","rb") as ds:
        test = pickle.load(ds)

else: 
    x,y = getDataset()

    train, val, test = getSets(x,y)

    with open("train.pkl","wb") as ds:
        (X_train, Y_train) = train
        Xtrain = []
        Ytrain = []
        for i in tqdm(range(len(X_train)), desc= "data augmentation"):
                Xtrain.append(data_augmentation(X_train[i]))
                Xtrain.append(X_train[i])
                Ytrain.append(Y_train[i])
                Ytrain.append(Y_train[i])

        X_train = np.array(Xtrain)
        Y_train = np.array(Ytrain)
        train = (X_train,Y_train)
        pickle.dump(train,ds)

    with open("val.pkl","wb") as ds:
        pickle.dump(val,ds)

    with open("test.pkl","wb") as ds:
        pickle.dump(test,ds)


(X_train, Y_train) = train
(X_val,Y_val) = val
(X_test, Y_test) = test





##TEST##


#for i in range(10):
#    cv2.imwrite(f"{i}.jpg",X_train[i])


##FINE TEST##

#A questo punto dobbiamo riconoscere i volti con una rete neurale

model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(64, 64, 3)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))

model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=[keras.metrics.CategoricalAccuracy()]
)


#model.summary()



if os.path.exists("pesiClassificatore.h5"):
    model.load_weights('pesiClassificatore.h5')

else:
    callback = keras.callbacks.EarlyStopping(monitor= "val_loss", patience=3)
    model.fit(
        X_train, to_categorical(Y_train), epochs=100, 
        batch_size=512, shuffle=True, 
        validation_data=(X_val,to_categorical(Y_val)),
        callbacks=callback
        )

    model.save_weights('pesiClassificatore.h5')

'''

results = model.evaluate(
  X_test,
  to_categorical(Y_test)
)
print(results)
'''

#qui dobbiamo ciclare ogni frame del video, per ogni frame ritagliare il volto
#passarlo alla rete neurale, farlo riconoscere e poi attaccare la label all'immagine

def classificatore(frames):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for f in tqdm(range(len(frames)), desc="Face recognition per frame"):
        faces,boxes = findFaces(frames[f])
        predict=model(faces)
        for (boxe,pred) in zip(boxes, predict):
            frames[f] = cv2.putText(frames[f], nomi[np.argmax(pred)] , (int(boxe.xyxy[0][0])-5,int(boxe.xyxy[0][1])-5),font, 1,(255,255,255),2)
            frames[f] = cv2.rectangle(frames[f], (int(boxe.xyxy[0][0]), int(boxe.xyxy[0][1])), (int(boxe.xyxy[0][2]), int(boxe.xyxy[0][3])), (255, 0, 255), 4)
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

results = classificatore(frames)
height, width, channels = results[0].shape
size = (width,height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out15 = cv2.VideoWriter('project_video_finale.mp4',fourcc, 15, size)

for i in tqdm(range(len(results)), desc="Saving frames into video"):
    out15.write(results[i])
out15.release()


def classificatoreIRT(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    faces,boxes = findFaces(frame)    
    predict=model(faces)
    try:
        for (boxe,pred) in zip(boxes, predict):
            frame = cv2.putText(frame, nomi[np.argmax(pred)] , (int(boxe.xyxy[0][0])-5,int(boxe.xyxy[0][1])-5),font, 1,(255,255,255),2)
            frame = cv2.rectangle(frame, (int(boxe.xyxy[0][0]), int(boxe.xyxy[0][1])), (int(boxe.xyxy[0][2]), int(boxe.xyxy[0][3])), (255, 0, 255), 4)
    except:
        pass    
    return frame
'''

camera = cv2.VideoCapture(0)
if not camera.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = camera.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    frame = classificatoreIRT(frame)
    cv2.imshow('Capture - Face detection', frame)
    if cv2.waitKey(10) == 27:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break

'''