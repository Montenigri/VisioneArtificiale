import numpy as np
import cv2
from datetime import datetime


src = 'rtsp://CV2023:Studente123@147.163.26.182:554/Streaming/Channels/101'
#src = 'rtsp://CV2023:Studente123@147.163.26.184:554/Streaming/Channels/101'
#video = cv2.VideoCapture(src)

ID = 0 # webcam
video = cv2.VideoCapture(src)
print(video)
if video is None or not video.isOpened():
    raise IOError("video not found")
else:
    print("frame width ", video.get(cv2.CAP_PROP_FRAME_WIDTH ))
    print("frame height ", video.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    print("frame rate ", video.get(cv2.CAP_PROP_FPS ))
    ret, img = video.read()
    key = ''
    key1 = ''
    while ret and key!=ord('q'):
        cv2.imshow("frame", img)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(f"test-{str(datetime.now().hour)+str(datetime.now().minute) + str(datetime.now().second)}.jpeg",img)
        ret, img = video.read()
        
    video.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
