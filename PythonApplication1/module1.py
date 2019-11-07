# -*- coding:utf-8 -*- 
import numpy as np
import imutils
import time
import cv2
#import V_Display as vd
#import V_UCom as com
import multiprocessing

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


#计算光流进程---占一个核
def Flow(frameQueue,flowX,flowY):
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
    while(True):
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
                        error = 1
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
                                error = 2
    #                        speed = speed[speed[:,2] >= ( max(speed[:,2])) ]
                        speed = speed[speed[:,2] <= (speed[filter_counter,2]) ]
                        FlowX = int(speed[0,0] * fps * 10)
                        FlowY = int(speed[0,1] * fps * 10)
                        filter_counter=0
                    else:
    #                        speed = speed[speed[:,2] == ( max(speed[:,2])) ]
                        FlowX = int(np.mean(speed[:,0]) * fps * 10)
                        FlowY = int(np.mean(speed[:,1]) * fps * 10)
#                    Uart_buf = bytearray([0x55,0xAA,0x20, (x  & 0xff00) >>8 , x & 0x00ff ,(y & 0xff00) >>8  , y  & 0x00ff ,0 ,0 ,0 ,0 , 0xAA])
#                    com.send(Uart_buf)
#                    print("speedX: %.1f ,speedY: %.1f"% (x,y))
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
            cv2.imshow("1",vis)

            fps_counter=fps_counter+1
            if (time.time() - start_time) > 1:
                fps=fps_counter / (time.time() - start_time)
                print("FPS: %.1f"%(fps))
                fps_counter = 0
                start_time = time.time()


            ch = cv2.waitKey(1)
            if ch == 27:
                break

            try:
                flowX.put(FlowX,False)
                flowY.put(FlowY,False)
            except:
                continue




if __name__ == '__main__':
    # 串口数据缓存
#    Uart_buf = [0x55,0xAA,0x30,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xAA]
    # 初始化串口
#    com.init(mode=2)
    #增加了底层驱动，可以直接通过cv2的0号设备读取摄像头
    cap = cv2.VideoCapture(0)
    cap.set(3,320)#设置摄像头输出宽
    cap.set(4,240)#设置摄像头输出高
#    print("start reading video...")
#    time.sleep(2.0)
#    print("start working")
    #初始化右侧图像显示功能,优化了传输,
    #增加了imshow('name',frame)函数，与cv2的imshow保持一致
    #无需指定fps
#    vd.init()
    #取一帧首图---为了后面光流的启动
    ret,firstFrame = cap.read()
    #创建主进程与光流进程之间的共享变量
    flowX=multiprocessing.Queue(maxsize=1)
    flowY=multiprocessing.Queue(maxsize=1)
    #创建图像队列
    frameQueue = multiprocessing.Queue(maxsize=1)
    #创建光流进程
    processFlow = multiprocessing.Process(target=Flow, args=(frameQueue,flowX,flowY))
    #运行光流进程
    processFlow.start()
    #初始化主线程中的光流变量
    xSpeed=0
    ySpeed=0
    #初始化主线程中的寻线变量
    # Black
    Lower = np.array([0, 0, 0])
    Upper = np.array([180, 255, 105])
    #Position
    Postion_x = 80
    Postion_y = 60
    angle =0 
    while(True):
#```````````````````获取图像``````````````````````
        ret,frame = cap.read()
#```````````````````向光流进程送入图像````````````
        try:
            frameQueue.put(frame,False)
        except:
            continue
#```````````````````处理寻线程序``````````````````
        frame = imutils.resize(frame, width=160)
        ROImask1 = frame[0:120,10:50]
        HSV1 = cv2.cvtColor(ROImask1, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(HSV1, Lower, Upper)
        mask1 = cv2.erode(mask1, None, iterations=2)
        mask1 = cv2.dilate(mask1, None, iterations=2)

        cnts1,h1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        cnts1 = cnts1[0] if imutils.is_cv2() else cnts1[1]


        ROImask2 = frame[0:120,60:100]
        HSV2 = cv2.cvtColor(ROImask2, cv2.COLOR_BGR2HSV)
        
        mask2 = cv2.inRange(HSV2, Lower, Upper)
        mask2 = cv2.erode(mask2, None, iterations=2)
        mask2 = cv2.dilate(mask2, None, iterations=2)

        cnts2,h2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        cnts2 = cnts2[0] if imutils.is_cv2() else cnts2[1]


        ROImask3 = frame[0:120,110:150]
        HSV3 = cv2.cvtColor(ROImask3, cv2.COLOR_BGR2HSV)
        
        mask3 = cv2.inRange(HSV3, Lower, Upper)
        mask3 = cv2.erode(mask3, None, iterations=2)
        mask3 = cv2.dilate(mask3, None, iterations=2)

        cnts3,h3 = cv2.findContours(mask3.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

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
#            if (rect2[0][0] > box2[0][0]) :
#                angle2 = int( - rect2[2] )
#            else :
#                angle2 = int( -90 - rect2[2])
            if(box2[1][1] >= box2[0][1]):
                angle2 = int( rect2[2] +180 )
            else:
                angle2 = int(rect2[2] * -1)
        else :
            angle2 =0
            Postion_x = 80
            Postion_y = 60
#```````````````````接收光流计算结果``````````````
        if not flowX.empty():
            xSpeed=flowX.get()
            ySpeed=flowY.get()
        print ("xSpeed: %d ,angle: %d ,lineY: %d"%(xSpeed,angle2,Postion_y))
#```````````````````处理串口数据并发送``````````````
#        Uart_buf = bytearray([0x55,0xAA,0x30,Postion_y>>8,Postion_y & 0x00ff, (angle2 & 0xff00)>>8,angle2 & 0x00ff,(xSpeed & 0xff00) >>8 , xSpeed & 0x00ff ,0x00 ,0x00 ,0xAA])
#        com.send(Uart_buf)
        cv2.imshow("flow",frame)

