import cv2 as cv
import numpy as np


def add_demo(m1,m2):
    dst=cv.add(m1,m2)
    cv.imshow("added image",dst)


def subtract_demo(m1,m2):
    dst=cv.subtract(m1,m2)
    cv.imshow("subtracted image",dst)


def divided_demo(m1,m2):
    dst=cv.divide(m1,m2)
    cv.imshow("divided image",dst)


def multiply_demo(m1,m2):
    dst=cv.multiply(m1,m2)
    cv.imshow("multiply image",dst)


def logic_demo(m1,m2):
    dst=cv.bitwise_and(m1,m2)    #逻辑与运算，只有两个二进制位都为1才是1，否则为0
    #dst=cv.bitwise_or(m1,m2)    #或运算，只有两个二进制位都为0才是0，否则为1
    #dst=cv.bitwise_not(m1)      #非运算，针对单个图像，按位取反，1为0，0为1
    #dst=cv.bitwise_xor(m1,m2)
    cv.imshow("logic image",dst)


def contrast_brightness_demo(image,c,b):    #图像对比度调节
    h,w,ch=image.shape
    blank=np.zeros([h,w,ch],image.dtype)
    dst=cv.addWeighted(image,c,blank,1-c,b)
    cv.imshow("con_bri_demo",dst)


def image_mean(m1,m2):
    M1,dev1=cv.meanStdDev(m1)
    M2,dev2=cv.meanStdDev(m2)
    print(M1)
    print(M2)
    print("=================")
    print(dev1)
    print(dev2)


src1 = cv.imread("1.jpg")
src2 = cv.imread("2.jpg")
#cv.namedWindow("image1",cv.WINDOW_AUTOSIZE)
cv.imshow("image1",src1)
#cv.imshow("image2",src2)

#add_demo(src1,src2)
#subtract_demo(src1,src2)
#divided_demo(src1,src2)
#multiply_demo(src1,src2)

#image_mean(src1,src2)

#logic_demo(src1,src2)

contrast_brightness_demo(src1,1.5,10)

cv.waitKey(0)
cv.destroyAllWindows()
