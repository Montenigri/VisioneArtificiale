###
#
# Questo script serve per annotare rapidamente sui frame i volti manualmente, in un array sono presenti i nomi dei presenti
# vengono quindi proposti man mano i volti e poi è possibile selezionare il nome corrispondente
#
###

import numpy as np
import cv2

dim = 64


faces_xml = 'haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(faces_xml):
    print('--(!)Error loading face cascade')
    exit(0)

def violajonesMultiple(img):
    faceROI = np.zeros((dim,dim), dtype=np.float32)
    faces = face_cascade.detectMultiScale(img)
    listOfFaceROI = []
    pos =[]
    for (x, y, w, h) in faces:
        faceROI = img[y:y + h, x:x + w]
        faceROI = cv2.resize(faceROI,(dim,dim), interpolation=cv2.INTER_LINEAR)
        listOfFaceROI.append(faceROI)
        pos.append([(x, y), (x+w, y+h), (255, 0, 255), 4])
    return listOfFaceROI, pos


video = cv2.VideoCapture("Da annotare a mano.mp4")
frames = []
count = 0
if (video.isOpened()== False):
    print("Error opening video file")
while(video.isOpened()):
    ret, frame = video.read()
    if ret == True and count<20:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            count+=1
    else:
        break

faces=[]
framesF = []
font = cv2.FONT_HERSHEY_SIMPLEX
nomi = ["Gabriele", "Stefano", "Davide", "Francesco"]
for frame in frames:
    vjMul, pos = violajonesMultiple(frame)
    for p in pos:
        px,py = p[0]
        frame = cv2.rectangle(frame, p[0],p[1],p[2],p[3])
        cv2.imshow("faccia", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(nomi)
        Nnome = int(input("inserisci indice del nome"))-1
        frame = cv2.putText(frame, nomi[Nnome], (px-5,py-5) ,font, 1,(255,255,255),2 )
    framesF.append(frame)

height, width = framesF[0].shape
size = (width,height)

fourcc = -1 

out30 = cv2.VideoWriter('project_annotato_a_mano.mp4',fourcc, 1, size)

for i in framesF:
    out30.write(i)
out30.release()