import cv2 as cv
from datetime import datetime
import time


src = 'rtsp://CV2023:Studente123@147.163.26.182:554/Streaming/Channels/101'
#src = 0

cap = cv.VideoCapture(src)
# Define the codec and create VideoWriter object

fourcc = cv.VideoWriter_fourcc(*'mp4v')
name = f"video-{str(datetime.now().hour)+str(datetime.now().minute) + str(datetime.now().second)}.mp4"
#La risoluzione deve essere uguale a quella della videocamera, altrimenti non salva nulla
out = cv.VideoWriter(name, fourcc, 20.0, (1280, 720))
framesRecorded = 0
font = cv.FONT_HERSHEY_SIMPLEX
if cap is None or not cap.isOpened():
    raise IOError("video not found")
else:
    print("frame width ", cap.get(cv.CAP_PROP_FRAME_WIDTH ))
    print("frame height ", cap.get(cv.CAP_PROP_FRAME_HEIGHT ))
    print("frame rate ", cap.get(cv.CAP_PROP_FPS ))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    out.write(frame)
    framesRecorded += 1
    cv.putText(frame,f'{framesRecorded}',(25,25), font, 1,(255,255,255),2,cv.LINE_AA)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q') or framesRecorded==40:
        break
# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()