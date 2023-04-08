import os
import glob 
import numpy as np
import cv2
from sklearn.decomposition import PCA


faces_xml = 'haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(faces_xml):
    print('--(!)Error loading face cascade')
    exit(0)


root = "foto64x64"
cwd = os.getcwd()
listDir = os.listdir(root)
tagFoto = {}

for dir in listDir:
    imgs =  glob.glob(f"{root}/{dir}/*.jpg")
    tagFoto[dir] = imgs



def violajones(frame):
    ListOfFaceROI = []
    faces = face_cascade.detectMultiScale(frame)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)   
        faceROI = frame[y:y + h, x:x + w]
        faceROI = faceROI.resize(64,64)
        ListOfFaceROI.append(faceROI)
    return ListOfFaceROI


#centro la faccia trovata tramite viola jones per avere un immagine centrata
def pad(array):
    x,y = np.shape(array)
    x1 = 0
    y1 = 0
    x = 64 - x
    y = 64 - y
    if x%2 == 1:
        x -=1
        x = x/2 
        x1 = x+1
    else:
        x = x/2
        x1 = x
    if y%2 == 1:
        y -=1
        y = y/2 
        y1 = y + 1 
    else:
        y = y/2
        y1 = y
    return np.pad(array , pad_width=((x, x1), (y,y1)), constant_values=0)

faceMean = np.zeros((64,64), dtype=np.float32)
faceNumber = 0

for key in tagFoto:
    for im in tagFoto[key]:
        img = cv2.imread(im,0)
        vj = violajones(img)
        for viola in vj:
            faceMean += violajones(viola)
            faceNumber+=1

faceMean /= faceNumber

faceMean = faceMean.astype(np.uint8)



cv2.imshow("viola" , faceMean)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
for key in tagFoto:
    newValue = []
    for im in tagFoto[key]:
        img = cv2.imread(im,0).astype(np.float32)
        eigenvalue= np.linalg.eig(img) 
        nv = [img, eigenvalue]
        newValue.append(nv)
    tagFoto.update({key : newValue})



listOfArray = []
for key in tagFoto:
    for im in tagFoto[key]:
        flat = cv2.imread(im,0).flatten()
        listOfArray.append(flat)

MatrixFlattenedImages = np.vstack(listOfArray)  

pca = PCA().fit(MatrixFlattenedImages)

n_components = 2000
eigenfaces = pca.components_[:n_components]

'''

