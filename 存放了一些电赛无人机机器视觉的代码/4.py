import cv2

flag=1

def cap2():
    cap=cv2.VideoCapture(1)
    while True:
        ret,frame=cap.read()
        cv2.imshow("1",frame)

if(flag==0)
    cap=cv2.VideoCapture(0)
    while True:
        ret,frame=cap.read()
        cv2.imshow("1",frame)
else:
    cap2()