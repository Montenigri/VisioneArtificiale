import os
import glob 
import numpy as np
import cv2
import time


start = time.time()

root = "foto64x64"
cwd = os.getcwd()
listDir = os.listdir(root)
tagFoto = {}

for dir in listDir:
    imgs =  glob.glob(f"{root}/{dir}/*.jpg")
    tagFoto[dir] = imgs


imgMean = np.zeros((64,64,3), dtype=np.float32)
imgNumber = 0

imgMeanPerPosition = []

for key in tagFoto:
    for im in tagFoto[key]:
        img = cv2.imread(im).astype(np.float32)
        imgMean += img
        imgNumber+=1
imgMean /= imgNumber

imgMean = imgMean.astype(np.uint8)

for key in tagFoto:
    number = 0
    newValue = []

    for im in tagFoto[key]:
        img = cv2.imread(im,0).astype(np.float32)
        eigenvalue= np.linalg.eig(img) 
        nv = [img, eigenvalue]
        newValue.append(nv)
        
    tagFoto.update({key : newValue})
startPrint = time.time()
print (tagFoto)
endPrint = time.time()

end = time.time()

timeElapsed = end - start
timePrint= endPrint-startPrint

print(f"tempo totale: {timeElapsed}")
print(f"tempo print: {timePrint}")

