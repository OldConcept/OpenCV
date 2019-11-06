# -*- coding:utf-8 -*-
import numpy as np
import imutils
import serial
import time
import cv2
import V_Display as vd
import math
import V_UCom as com

# Send_data
Uart_buf = [0x55, 0xAA, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0xAA]
# Setup Usart
com.init(mode=2)

# Black
Lower = np.array([0, 0, 0])
Upper = np.array([180, 255, 100])

# Position
Postion_x = 80
Postion_y = 60
angle = 0

# 增加了底层驱动，可以直接通过cv2的0号设备读取摄像头
cap = cv2.VideoCapture(0)
cap.set(3, 320)  # 设置摄像头输出宽
cap.set(4, 240)  # 设置摄像头输出高
print("start reading video...")
time.sleep(2.0)
print("start working")
# 初始化右侧图像显示功能,优化了传输,
# 增加了imshow('name',frame)函数，与cv2的imshow保持一致
# 无需指定fps
vd.init()

while (True):
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=160)
    ROImask1 = frame[0:120, 10:50]
    HSV1 = cv2.cvtColor(ROImask1, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(HSV1, Lower, Upper)
    mask1 = cv2.erode(mask1, None, iterations=2)
    mask1 = cv2.dilate(mask1, None, iterations=2)

    cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts1 = cnts1[0] if imutils.is_cv2() else cnts1[1]

    ROImask2 = frame[0:120, 60:100]
    HSV2 = cv2.cvtColor(ROImask2, cv2.COLOR_BGR2HSV)

    mask2 = cv2.inRange(HSV2, Lower, Upper)
    mask2 = cv2.erode(mask2, None, iterations=2)
    mask2 = cv2.dilate(mask2, None, iterations=2)

    cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts2 = cnts2[0] if imutils.is_cv2() else cnts2[1]

    ROImask3 = frame[0:120, 110:150]
    HSV3 = cv2.cvtColor(ROImask3, cv2.COLOR_BGR2HSV)

    mask3 = cv2.inRange(HSV3, Lower, Upper)
    mask3 = cv2.erode(mask3, None, iterations=2)
    mask3 = cv2.dilate(mask3, None, iterations=2)

    cnts3 = cv2.findContours(mask3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts3 = cnts3[0] if imutils.is_cv2() else cnts3[1]

    if (len(cnts2) > 0):
        if(len(cnts1) >0):
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
        if(len(cnts3)>0):
            c3 = max(cnts3, key=cv2.contourArea)

            rect3 = cv2.minAreaRect(c3)
            box3 = np.int0(cv2.boxPoints(rect3))
            box3[0][0] = box3[0][0] + 110
            box3[1][0] = box3[1][0] + 110
            box3[2][0] = box3[2][0] + 110
            box3[3][0] = box3[3][0] + 110
            cv2.drawContours(frame, [box3], 0, (0, 0, 255), 2)

        Postion_x = int(rect2[0][0])
        Postion_y = int(rect2[0][1])
        if (rect2[0][0] > box2[0][0]):
            angle2 = int(- rect2[2])
        else:
            angle2 = int(-90 - rect2[2])
    else:
        angle2 = 0
        Postion_x = 80
        Postion_y = 60

    print(angle, Postion_y)
    Uart_buf = bytearray(
        [0x55, 0xAA, 0x30, Postion_y >> 8, Postion_y & 0x00ff, (angle2 & 0xff00) >> 8, angle & 0x00ff, 0, 0, 0, 0,
         0xAA])
    com.send(Uart_buf)
    vd.show(frame)
