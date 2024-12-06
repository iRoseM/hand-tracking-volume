import time
import numpy as np
import cv2
import HandTrackingModule as htm
import math
# img= cv2.imread('assets/pfp.jpg', 1) #-1 for colored pic, 0 for grey color, 1 for alpha
# img= cv2.resize(img, (400,400))
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


wCam, hCam= 460, 480
prevTime= 0

detector= htm.handDetector(detectionCon= 0.7)

cap= cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
while True:
    ret, frame= cap.read() #return image frame (ret is return)
    frame= detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    if len(lmList) != 0:
        print(lmList[4], lmList[8])

    currTime= time.time()
    fps= 1/(currTime - prevTime)
    prevTime = currTime

    cv2.putText(frame, f'FPS: {int(fps)}', (40,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0 , 0), 3)

    cv2.imshow('frame', frame)

    if(cv2.waitKey(1) == ord('q')): #if q pressed the frame is stoped
        break

cap.release()
cv2.destroyAllWindows()