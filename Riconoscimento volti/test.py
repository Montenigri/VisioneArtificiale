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
faceLabel = []

for key in tagFoto:
    for im in tagFoto[key]:
        vj = violajones(im)
        faceMean += vj
        flat = vj.flat
        listOfArray.append(flat)
        faceLabel.append(key)
        faceNumber+=1

faceMean /= faceNumber

#faceMean = faceMean.astype(np.uint8)
MatrixFlattenedImages = np.vstack(listOfArray)  
flattenFaceMean = faceMean.flatten()

'''
pca = PCA().fit(MatrixFlattenedImages)


with open('pca.pkl', 'wb') as pickle_file:
        pickle.dump(pca, pickle_file)

'''

with open('pca.pkl', 'rb') as pickle_file:
    pca = pickle.load(pickle_file)

n_components = 4096
eigenfaces = pca.components_[:n_components]
 
# Show the first 16 eigenfaces
#fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
#for i in range(16):
#    axes[i%4][i//4].imshow(eigenfaces[i].reshape(64,64), cmap="gray")
#plt.show()


if "Gabriele_Musso" in faceLabel:
  print("Element is in the list")
else:
  print("Element is not in the list")

weights = eigenfaces @ (MatrixFlattenedImages - pca.mean_).T

vj = violajones("test/0.jpg")
query = vj.reshape(1,-1)
query_weight = eigenfaces @ (query - pca.mean_).T
euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
best_match = np.argmin(euclidean_distance)
print("Best match %s with Euclidean distance %f" % (faceLabel[best_match], euclidean_distance[best_match]))
# Visualize
fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
axes[0].imshow(query.reshape(64,64), cmap="gray")
axes[0].set_title("Query")
axes[1].imshow(MatrixFlattenedImages[best_match].reshape(64,64), cmap="gray")
axes[1].set_title(f"{faceLabel[best_match]} distanza: {euclidean_distance[best_match]}")
plt.show()


'''
Training violajons con una sola foto, poi ricopio la funzione ed aggiungo un array da tornare
a quel punto cerco il best match
'''