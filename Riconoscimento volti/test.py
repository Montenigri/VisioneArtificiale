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
#"Gabriele", "Stefano", "Davide", "Francesco",
names = ["Francesco", "Stefano", "Davide","Francesco", "Stefano", "Davide","Francesco", "Stefano", "Davide","Francesco", "Stefano", "Davide","Francesco", "Stefano", "Davide","Francesco", "Stefano", "Davide","Francesco", "Stefano", "Davide","Francesco", "Stefano", "Davide","Francesco", "Stefano", "Davide","Francesco", "Davide", "Stefano","Francesco", "Davide", "Stefano","Francesco", "Davide", "Stefano","Davide","Francesco", "Stefano","Davide","Francesco", "Stefano","Davide","Francesco", "Stefano","Davide","Francesco", "Stefano", "Francesco","Davide","Stefano","Davide","Francesco","Stefano","Davide","Francesco","Stefano","Davide","Francesco","Gabriele","Stefano","Davide","Francesco","Stefano", "Gabriele", "Francesco", "Davide","Stefano","Davide","Francesco","Stefano","Gabriele","Davide","Francesco","Stefano","Gabriele"]
for frame in frames:
    vjMul, pos = violajonesMultiple(frame)
    for p in pos:
        px,py = p[0]
        frame = cv2.putText(frame, names[Count], (px-5,py-5) ,font, 1,(255,255,255),2 )
        Count+=1
        frame = cv2.rectangle(frame, p[0],p[1],p[2],p[3])
    framesF.append(frame)

faces = [item for sublist in faces for item in sublist]


Count = 0
for face in framesF:
    cv2.imwrite(f"Annotate/frame{Count}.jpg",face)
    Count+=1