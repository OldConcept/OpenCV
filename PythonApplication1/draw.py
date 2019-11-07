import cv2
import numpy as np

drawing=False
mode=True
ix,iy=-1,-1

def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y
    elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
        if drawing==True:
            if mode==True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv2.circle(img,(x,y),10,(0,0,255),-1)
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False


img=np.zeros((400,400,3),np.uint8)
cv2.namedWindow("0")

cv2.ellipse(img,(200,200),(100,80),90,0,275,(255,0,0),1)
cv2.line(img,(100,20),(390,390),(0,255,0),2)
cv2.putText(img,"Hello",(10,390),cv2.FONT_HERSHEY_SIMPLEX,4,(0,0,255))

cv2.setMouseCallback("0",draw_circle)

while(1):
    cv2.imshow("0",img)
    if cv2.waitKey(1)==ord("m"):
        mode=not mode
    elif cv2.waitKey(1)==27:
        break

cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,)
cv2.destroyAllWindows()