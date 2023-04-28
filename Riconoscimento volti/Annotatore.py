import os
import glob 
import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle


faces_xml = 'haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(faces_xml):
    print('--(!)Error loading face cascade')
    exit(0)

def violajonesMultiple(img):
    faceROI = np.zeros((64,64), dtype=np.float32)
    faces = face_cascade.detectMultiScale(img,minNeighbors=5, minSize = (10,10), maxSize=(50,50))
    listOfFaceROI = []
    pos =[]
    for (x, y, w, h) in faces:
        faceROI = img[y:y + h, x:x + w]
        faceROI = cv2.resize(faceROI,(64,64), interpolation=cv2.INTER_LINEAR)
        listOfFaceROI.append(faceROI)
        pos.append([(x, y), (x+w, y+h), (255, 0, 255), 4])
    return listOfFaceROI, pos


video = cv2.VideoCapture("Da annotare a mano.mp4")
frames = []
if (video.isOpened()== False):
    print("Error opening video file")
while(video.isOpened()):
    ret, frame = video.read()
    if ret == True:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    else:
        break

Count = 0
faces=[]
framesF = []
font = cv2.FONT_HERSHEY_SIMPLEX
nomi = ["Gabriele", "Stefano", "Davide", "Francesco"]
names = []
for frame in frames:
    vjMul, pos = violajonesMultiple(frame)
    for p in pos:
        px,py = p[0]
        Count+=1
        frame = cv2.rectangle(frame, p[0],p[1],p[2],p[3])

        cv2.imshow("faccia", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(nomi)
        Nnome = int(input("inserisci indice del nome"))
        names.append(nomi[Nnome])
        frame = cv2.putText(frame, nomi[Nnome], (px-5,py-5) ,font, 1,(255,255,255),2 )
        

    framesF.append(frame)

Count = 0
for face in framesF:
    cv2.imwrite(f"Annotate/frame{Count}.jpg",face)
    Count+=1