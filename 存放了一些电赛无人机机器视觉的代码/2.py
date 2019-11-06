# -*- coding:utf-8 -*- 
import numpy as np
import imutils
import time
import cv2
import pyzbar.pyzbar as pyzbar
import V_Display as vd
import V_UCom as com
import sys
import multiprocessing

reload(sys)
sys.setdefaultencoding('utf8')

photoTimes=0

def decodeDisplay(image):
    img=image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    global photoTimes
    barcodes = pyzbar.decode(gray)
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        if( len(barcodes)>0 and photoTimes!=3 and barcodeType=="QRCODE"):
            cv2.imwrite('/boot/'+str(photoTimes)+'QR.jpg',img)
            photoTimes = photoTimes+1
        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,.5, (0, 0, 125), 2)
        #print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData)) 
    #vd.show(image)


def Black(frame):
    fps_counter=0
    start_time=time.time()
    frame = imutils.resize(frame, width=160)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

# Apply edge detection method on the image
#        edges = cv2.Canny(gray,1,250)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    kernel = np.ones((5,5),np.uint8) 
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges2 = cv2.Canny(closing,50,150,apertureSize = 3)
    edges3 = cv2.cvtColor(edges2,cv2.COLOR_GRAY2BGR)
    erosion = cv2.dilate(edges3,kernel,iterations = 1)

    fps_counter=fps_counter+1
    if (time.time() - start_time) > 1:
        fps=fps_counter / (time.time() - start_time)
        print("FPS: %.1f"%(fps))
        fps_counter = 0
        start_time = time.time()

    return erosion
           
#
#            ch = cv2.waitKey(1)
#            if ch == 27:
#                break

if __name__ == '__main__':
    # 串口数据缓存
    Uart_buf = [0x55,0xAA,0x60,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xAA]
    # 初始化串口
    com.init(mode=2)
    #增加了底层驱动，可以直接通过cv2的0号设备读取摄像头
    cap = cv2.VideoCapture(0)
    cap.set(3,160)#设置摄像头输出宽
    cap.set(4,120)#设置摄像头输出高
    camera=cv2.VideoCapture(1)
    camera.set(3,480)
    camera.set(4,360)
    print("start reading video...")
    time.sleep(2.0)
    print("start working")
    #初始化右侧图像显示功能,优化了传输,
    #增加了imshow('name',frame)函数，与cv2的imshow保持一致
    #无需指定fps
    vd.init()
    Lower = np.array([0, 0, 0])
    Upper = np.array([180, 255, 105])
    #Position
    Postion_x = 80
    Postion_y = 60
    angle =0 
    while(True):
#```````````````````获取图像``````````````````````
        ret,frame = cap.read()
        ret1,frame1 =camera.read()

        decodeDisplay(frame1)
        frame = Black(frame)

#            H = frame.shape[0]
#            W = frame.shape[1]
#
#            for i in range(H):
#                for j in range(W):
#                    frame[i,j] = 255 - frame[i,j]
        frame = 255 - frame
#```````````````````处理寻线程序``````````````````
        frame = imutils.resize(frame, width=160)
        ROImask1 = frame[0:120,10:50]
        HSV1 = cv2.cvtColor(ROImask1, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(HSV1, Lower, Upper)
        mask1 = cv2.erode(mask1, None, iterations=2)
        mask1 = cv2.dilate(mask1, None, iterations=2)

        cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        cnts1 = cnts1[0] if imutils.is_cv2() else cnts1[1]


        ROImask2 = frame[0:120,60:100]
        HSV2 = cv2.cvtColor(ROImask2, cv2.COLOR_BGR2HSV)
        
        mask2 = cv2.inRange(HSV2, Lower, Upper)
        mask2 = cv2.erode(mask2, None, iterations=2)
        mask2 = cv2.dilate(mask2, None, iterations=2)

        cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        cnts2 = cnts2[0] if imutils.is_cv2() else cnts2[1]


        ROImask3 = frame[0:120,110:150]
        HSV3 = cv2.cvtColor(ROImask3, cv2.COLOR_BGR2HSV)
    
        mask3 = cv2.inRange(HSV3, Lower, Upper)
        mask3 = cv2.erode(mask3, None, iterations=2)
        mask3 = cv2.dilate(mask3, None, iterations=2)

        cnts3 = cv2.findContours(mask3.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        cnts3 = cnts3[0] if imutils.is_cv2() else cnts3[1]



        if (len(cnts1) > 0 and len(cnts2) > 0  and len(cnts3) > 0) :
            c1 = max(cnts1, key=cv2.contourArea)

            rect1 = cv2.minAreaRect(c1)
            box1 = np.int0(cv2.boxPoints(rect1))
            box1[0][0] = box1[0][0] 
            box1[1][0] = box1[1][0] 
            box1[2][0] = box1[2][0] 
            box1[3][0] = box1[3][0] 
            cv2.drawContours(frame, [box1], 0, (0, 0, 255), 2)


            c2 = max(cnts2, key=cv2.contourArea)

            rect2 = cv2.minAreaRect(c2)
            box2 = np.int0(cv2.boxPoints(rect2))
            box2[0][0] = box2[0][0] + 60
            box2[1][0] = box2[1][0] + 60
            box2[2][0] = box2[2][0] + 60
            box2[3][0] = box2[3][0] + 60
            cv2.drawContours(frame, [box2], 0, (0, 0, 255), 2)


            c3 = max(cnts3, key=cv2.contourArea)

            rect3 = cv2.minAreaRect(c3)
            box3 = np.int0(cv2.boxPoints(rect3))
            box3[0][0] = box3[0][0] + 110
            box3[1][0] = box3[1][0] + 110
            box3[2][0] = box3[2][0] + 110
            box3[3][0] = box3[3][0] + 110
            cv2.drawContours(frame, [box3], 0, (0, 0, 255), 2)

            Postion_x = int( rect2[0][0] )
            Postion_y = int( rect2[0][1] )

            if(box2[1][1] >= box2[0][1]):
                angle = int( rect1[2] +90 )
            else:
                angle = int(rect1[2] * -1)
        
        else :
            angle =0
            Postion_x = 80
            Postion_y = 60

#```````````````````处理串口数据并发送``````````````
        Uart_buf = bytearray([0x55,0xAA,0x30,Postion_y>>8,Postion_y & 0x00ff, (angle & 0xff00)>>8,angle & 0x00ff,0x00 >>8 ,0x00,0x00,0x00,0xAA])
        com.send(Uart_buf)
        vd.show(frame)