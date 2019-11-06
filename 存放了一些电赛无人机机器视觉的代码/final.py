# -*- coding:utf-8 -*- 
import numpy as np
import imutils
import time
import cv2
import V_Display as vd
import V_UCom as com
import sys
import pyzbar.pyzbar as pyzbar
import multiprocessing
import math


reload(sys)  
sys.setdefaultencoding('utf8')

QRPhotoTimes=0
TXPhotoTimes=0

# Send_data
Uart_buf = [0x55,0xAA,0x00,0x00,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0xAA]
    
def Blob():
    cap = cv2.VideoCapture(0)
    cap.set(3,160)#设置摄像头输出宽
    cap.set(4,120)#设置摄像头输出高
    print("start reading video...")
    time.sleep(2.0)
    print("start working")
    
    t0=0
    t1=0
    flag=0
    STOP=0
# Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
    params.minThreshold = 80
    params.maxThreshold = 140
    params.filterByArea = True
    params.minArea = 10
        # Filter by Color.
    params.filterByColor = True
    params.blobColor = 0
        # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
    else:
            detector = cv2.SimpleBlobDetector_create(params)
    
    while(True):
        ret,frame = cap.read()
        frame = imutils.resize(frame, width=160)
        keypoints = detector.detect(frame)
        if (keypoints):
            z=len(keypoints)
            area=list(range(0,z))
            for i in range(z):
                area[i]=keypoints[i].size
                if(int(area[i-1])>int(area[i])):
                    x=keypoints[i-1].pt[0]
                    y=keypoints[i-1].pt[1]
                    i=i-1
                else:
                    x=keypoints[i].pt[0]
                    y=keypoints[i].pt[1]
            Postion_x = int(x)#取整
            Postion_y = int(y)
            cv2.circle(frame, (Postion_x ,Postion_y), 3, (255, 255, 0),-1)  # 标记圆心，这里的参数值都必须为整数,-1表示把圆填满
        else:
            Postion_x = 80
            Postion_y = 60
        if flag==0 and Postion_x!=80:
            t0=time.clock()
            flag=1
        t1=time.clock()
        if int(t1-t0)>4 and t0!=0:
            STOP=1
        print('circle_center:',Postion_x,Postion_y,t1-t0,STOP)
        STOP=0
        flag=1
        Uart_buf = bytearray([0x55,0xAA,0x10,Postion_x>>8,Postion_x & 0x00ff,
                            Postion_y>>8,Postion_y & 0x00ff,0x00,0x00,flag,STOP,0xAA])
        com.send(Uart_buf)
        '''im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (255, 255, 255),
                                                cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        vd.show(im_with_keypoints)'''

def decodeDisplay(Flag):#找二维码和条形码开一个核
    global QRPhotoTimes
    global TXPhotoTimes
    fpsCounter=0

    flag=0
    camera=cv2.VideoCapture(1)
    camera.set(3,480)
    camera.set(4,360)
    
    lower_yellow = np.array([22, 50, 50])
    upper_yellow = np.array([34, 255, 255])

    while(True):        
        ret,image=camera.read()
        fpsCounter=fpsCounter+1
        if(fpsCounter==31):
            fpsCounter=0
            flag=0
        img=image.copy()
        frame = imutils.resize(image, width=160)
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask_yellow=cv2.inRange(hsv,lower_yellow,upper_yellow)
        output=cv2.bitwise_and(hsv,hsv,mask=mask_yellow)
        output=cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
        cnts = cv2.findContours(output.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts=cnts[0] if imutils.is_cv2() else cnts[1]
        if(len(cnts)>0):
            fpsCounter=0
            #cnt=max(cnts,key=cv2.contourArea)
            #rect=cv2.minAreaRect(cnt)
            #box=np.int0(cv2.boxPoints(rect))
            #cv2.drawContours(image,[box],0,(0,0,255),2)
            flag=1
            if(TXPhotoTimes!=3):
                cv2.imwrite('/boot/'+str(TXPhotoTimes)+'TX.jpg',img)
                TXPhotoTimes=TXPhotoTimes+1

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)
        for barcode in barcodes:
            #(x, y, w, h) = barcode.rect
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            if( len(barcodes)>0 and QRPhotoTimes!=3 and barcodeType=="QRCODE"):
                cv2.imwrite('/boot/'+str(QRPhotoTimes)+'QR.jpg',img)
                QRPhotoTimes = QRPhotoTimes+1
            '''text = "{} ({})".format(barcodeData, barcodeType)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,.5, (0, 0, 125), 2)
            print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData)) '''
        #print("条形码："+str(TXPhotoTimes)+","+"二维码："+str(QRPhotoTimes))
        vd.show(image)
        try:
            Flag.put(flag,False)
        except:
            continue

def Flow(frameQueue,flowX,flowY):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 2,
            qualityLevel = 0.5,
            minDistance = 7,
            blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (11,11),
        maxLevel = 2,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),minEigThreshold = 1e-4)

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    track_len = 5
    detect_interval = 5
    tracks = []
    frame_idx = 0
    fps_counter = 0
    start_time = time.time()
    fps=0
    filter_counter = 0
    FlowX =0 
    FlowY =0 
    while True:
        if not frameQueue.empty():
            frame = frameQueue.get()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(tracks) > 0:
                img0, img1 = prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                tracks = new_tracks
                cnt = len(tracks)
                speed=np.zeros((cnt,3))
                for i in range(0,len(tracks)):
                    try:
                        speed[i][0] = tracks[i][1][0] - tracks[i][0][0]
                        speed[i][1] = tracks[i][1][1] - tracks[i][0][1]
                        speed[i][2] = math.sqrt(speed[i][0]*speed[i][0] + speed[i][1]*speed[i][1])
                    except:
                        continue
                speed = speed[np.lexsort(-speed.T)]
                if(fps !=None and len(tracks) !=0):
                    if( len(tracks)> 3):
                #                        print(len(self.tracks))
                #                        print(len(speed)) 
                        for i in range(0,len(tracks)-1):
                            try:
                #                            print (speed[i][2] ,speed[i+1][2] ) 
                                if(speed[i][2]*0.9 >= speed[i+2][2]):
                #                                    speed = np.delete(speed, 0, 0)
                                    filter_counter = filter_counter + 1
                                else:
                                    break
                            except:
                                continue
                #                        speed = speed[speed[:,2] >= ( max(speed[:,2])) ]
                        speed = speed[speed[:,2] <= (speed[filter_counter,2]) ]
                        FlowX = int(speed[0,0] * fps * 10)
                        FlowY = int(speed[0,1] * fps * 10)
                        filter_counter=0
                    else:
                #                        speed = speed[speed[:,2] == ( max(speed[:,2])) ]
                        FlowX = int(np.mean(speed[:,0]) * fps * 10)
                        FlowY = int(np.mean(speed[:,1]) * fps * 10)
                    #print("speedX: %.1f ,speedY: %.1f"% (x,y))
                cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))

            if frame_idx % detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        tracks.append([(x, y)])
                    if(len(tracks)>20):
                        tracks = np.delete(tracks, [0,1], axis=0)


            frame_idx += 1
            prev_gray = frame_gray
            #vd.show(vis)

            fps_counter=fps_counter+1
            if (time.time() - start_time) > 1:
                fps=fps_counter / (time.time() - start_time)
                print("FPS: %.1f"%(fps))
                fps_counter = 0
                start_time = time.time()

            try:
                flowX.put(FlowX,False)
                flowY.put(FlowY,False)
            except:
                continue

def Black(frame):
    '''fps_counter=0
    start_time=time.time()'''
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

    '''fps_counter=fps_counter+1
    if (time.time() - start_time) > 1:
        fps=fps_counter / (time.time() - start_time)
        print("FPS: %.1f"%(fps))
        fps_counter = 0
        start_time = time.time()'''

    return erosion

def BlackAndFind():
    cap = cv2.VideoCapture(0)
    cap.set(3,320)#设置摄像头输出宽
    cap.set(4,240)#设置摄像头输出高
    print("start reading video...")
    time.sleep(2.0)
    print("start working")

    ret,firstFrame = cap.read()

    xSpeed=0
    ySpeed=0
    buzz_flag=0

    flowX=multiprocessing.Queue(maxsize=1)
    flowY=multiprocessing.Queue(maxsize=1)
    frameQueue = multiprocessing.Queue(maxsize=1)
    processFlow = multiprocessing.Process(target=Flow, args=(frameQueue,flowX,flowY))
    processFlow.start()

    Flag=multiprocessing.Queue(maxsize=1)
    processDecode=multiprocessing.Process(target=decodeDisplay,args=(Flag,))
    processDecode.start()
    
    Lower = np.array([0, 0, 0])
    Upper = np.array([180, 255, 105])

    fps_counter=0
    start_time=time.time()

    while(True):
        ret,frame = cap.read()
        try:
            frameQueue.put(frame,False)
        except:
            continue
        frame = Black(frame)
        frame = 255 - frame
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
        
        fps_counter=fps_counter+1
        if (time.time() - start_time) > 1:
            fps=fps_counter / (time.time() - start_time)
            print("FPS: %.1f"%(fps))
            fps_counter = 0
            start_time = time.time()

        if not flowX.empty():
            xSpeed=flowX.get()
            ySpeed=flowY.get()
        print ("xSpeed: %d"%(xSpeed))
        if not Flag.empty():
            buzz_flag=Flag.get()
        print(buzz_flag)
#```````````````````处理串口数据并发送``````````````
        Uart_buf = bytearray([0x55,0xAA,0x30,Postion_y>>8,Postion_y & 0x00ff,0x00,0x00,(xSpeed & 0xff00) >>8 , xSpeed & 0x00ff ,0x00,buzz_flag,0xAA])
        com.send(Uart_buf)
        #vd.show(frame)



if __name__ == '__main__':
    # Setup Usart
    com.init(mode=2)
   #增加了底层驱动，可以直接通过cv2的0号设备读取摄像头
    vd.init()
    #FLAG=com.Rev()
    FLAG=1

    if(FLAG==0):
        Blob()
    elif(FLAG==1):
        BlackAndFind()
    else:
        while(True):
            angle=0
            Postion_x = 80
            Postion_y = 60
            Uart_buf = bytearray([0x55,0xAA,0x30,Postion_y>>8,Postion_y & 0x00ff,0x00,0x00,0x00,0x00,0x00,0x00,0xAA])
            com.send(Uart_buf)
            