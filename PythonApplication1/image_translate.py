import cv2 as cv
import numpy as np


def extract_video():
    capture=cv.VideoCapture("22.mp4")
    #capture.open("http://admin:admin@192.168.1.100:8081")
    while(True):
        ret,frame=capture.read()
        if ret==False:
            break
        hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        lower_hsv=np.array([0,0,221])
        upper_hsv=np.array([180,30,255])
        mask=cv.inRange(hsv,lowerb=lower_hsv,upperb=upper_hsv)
        dst=cv.bitwise_and(frame,frame,mask=mask)
        cv.imshow("frame",frame)
        cv.imshow("mask",mask)
        cv.imshow("changed video",dst)
        if cv.waitKey(100)==27:
            break


def image_translate(image):
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)#将RGB转换为GRAY
    cv.imshow("gray",gray)
    hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)#将RGB转换为HSV    H:0-180  S:0-255  V:0-255
    cv.imshow("hsv",hsv)
    yuv=cv.cvtColor(image,cv.COLOR_BGR2YUV)#将RGB转换为YUV
    cv.imshow("yuv",yuv)
    Ycrcb=cv.cvtColor(image,cv.COLOR_BGR2YCrCb)#将RGB转换为Ycrcb
    cv.imshow("Ycrcb",Ycrcb)


#src=cv.imread("1.jpg")
#cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#cv.imshow("input image",src)

#b,g,r=cv.split(src)     #图像通道分离
#cv.imshow("blue",b)
#cv.imshow("green",g)
#cv.imshow("red",r)

#src[:,:,2]=0      #矩阵对应的分别是图片三通道B,G,R
#src=cv.merge([b,g,r])
#cv.imshow("changed image",src)

#image_translate(src)
extract_video()
cv.waitKey(0)
cv.destroyAllWindows()