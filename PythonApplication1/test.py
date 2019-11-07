import cv2 as cv
import numpy as np


def access_pixels(image):
    print(image.shape)
    height=image.shape[0]
    width=image.shape[1]
    channels=image.shape[2]
    print("width: %s,height: %s,channels :%s"%(width,height,channels))
    for row in range(height):
        for rank in range(width):
            for high in range(channels):
                pv=image[row,rank,high]
                image[row,rank,high]=255-pv    #对图像的numpy数组值进行修改
    cv.imshow("pixel image",image)
    

def video_demo():
    capture=cv.VideoCapture(0)
    while(True):
        ret,frame=capture.read()
        frame=cv.flip(frame,1)
        cv.imshow("video",frame)
        c=cv.waitKey(100)
        if c==27:    #ESC键对应的ASCII码值为27，按下ESC键循环中止
            break


def get_image_info(image):
    print(type(image))#
    print(image.shape)#显示图片（宽、高、通道数）
    print(image.size)#显示图片（宽*高*通道数）
    print(image.dtype)#
    pixel_data=np.array(image)
    print(pixel_data)


src = cv.imread("1.jpg")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
#gray=cv.cvtColor(src,cv.COLOR_RGBA2GRAY)    #将图像转为灰度图
#get_image_info(src)
#access_pixels(src)
video_demo()
#access_pixels(src)
cv.waitKey(0)
cv.destroyAllWindows()