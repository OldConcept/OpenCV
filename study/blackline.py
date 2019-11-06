import numpy as np
import cv2 as cv

cap=cv.VideoCapture(0)

while True:
    ret,frame=cap.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    ret,thresh=cv.threshold(gray,127,255,0)
    contours,hierarchy=cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cnt=contours[0]
    c=max(cnt,key=cv.contourArea)
    rect=cv.minAreaRect(c)
    box1 = np.int0(cv.boxPoints(rect))
    cv.drawContours(frame, [box1], 0, (0, 0, 255), 2)
    cv.imshow("rect",frame)
    print("("+str(box1[0][0])+","+str(box1[0][1])+")"+"("+str(box1[1][0])+","+str(box1[1][1])+")"+"("+str(box1[2][0])+","+str(box1[2][1])+")"+
          "("+str(box1[3][0])+","+str(box1[3][1])+")")
