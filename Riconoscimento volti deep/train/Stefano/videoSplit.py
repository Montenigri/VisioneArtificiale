

#dividere i video in frame
import cv2

video = cv2.VideoCapture("video-114313.mp4")
frames = []
if (video.isOpened()== False):
    print("Error opening video file")
while(video.isOpened()):
    ret, frame = video.read()
    if ret == True:
            frames.append(frame)
    else:
        break


Count = 0
for frame in frames:
    cv2.imwrite(f"Stefano_{Count}.jpg",frame)
    Count+=1     