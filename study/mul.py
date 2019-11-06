# -*- coding:utf-8 -*-
import cv2
import pyzbar.pyzbar as pyzbar
import time
#import V_Display as vd
import imutils
#import sys

#reload(sys)
#sys.setdefaultencoding('utf8')

#vd.init()
camera = cv2.VideoCapture(0)


def decodeDisplay(image):
    img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    global photoTimes
    barcodes = pyzbar.decode(gray)
    for barcode in barcodes:
        # 提取二维码的边界框的位置
        # 画出图像中条形码的边界框
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 提取二维码数据为字节对象，所以如果我们想在输出图像上
        # 画出来，就需要先将它转换成字符串
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        # 绘出图像上条形码的数据和条形码类型
        '''if (len(barcodes) > 0 and photoTimes != 3 and barcodeType == "QRCODE"):
            cv2.imwrite('/boot/' + str(photoTimes) + 'QR.jpg', img)
            photoTimes = photoTimes + 1'''
        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 125), 2)
        # 向终端打印条形码数据和条形码类型
        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
    #vd.show(image)
    cv2.imshow("QR",image)


#camera.set(3, 640)  # 设置分辨率
#camera.set(4, 480)  # 设置分辨率
photoTimes = 0
# time.sleep(0.5)
while (True):
    # 读取当前帧
    ret, frame = camera.read()
    # frame = imutils.resize(frame, width=160)
    decodeDisplay(frame)
