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


def violajones(im):
    img = cv2.imread(im,0)
    faceROI = np.zeros((64,64), dtype=np.float32)
    faces = face_cascade.detectMultiScale(img)
    for (x, y, w, h) in faces:
        faceROI = img[y:y + h, x:x + w]
        faceROI = cv2.resize(faceROI,(64,64), interpolation=cv2.INTER_LINEAR)
        
    return faceROI


#centro la faccia trovata tramite viola jones per avere un immagine centrata

faceMean = np.zeros((64,64), dtype=np.float32)
faceNumber = 0
listOfArray = []


for key in tagFoto:
    for im in tagFoto[key]:
        vj = violajones(im)
        faceMean += vj
        flat = vj.flat
        listOfArray.append(flat)
        faceNumber+=1

faceMean /= faceNumber

#faceMean = faceMean.astype(np.uint8)
MatrixFlattenedImages = np.vstack(listOfArray)  
flattenFaceMean = faceMean.flatten()

covArray = []

for key in tagFoto:
    for im in tagFoto[key]:
        vj = violajones(im)
        flat = vj.flatten()
        covArray.append(flat-flattenFaceMean)
'''
def covSub(_1d):
    return _1d-flattenFaceMean

covArray = np.apply_along_axis(covSub,1,MatrixFlattenedImages)
'''

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

covArray = np.array(covArray)
cov = np.dot(np.transpose(covArray),covArray)
cov /= faceNumber






#cov = np.cov(MatrixFlattenedImages, faceMean)

#cv2.imshow("viola" , faceMean)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#print(cov)




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

