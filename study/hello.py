# -*- coding:utf-8 -*- 
import numpy as np
import imutils
import time
import cv2
# import V_Display as vd
# import V_UCom as com
import math

# Send_data
# Uart_buf = [0x55,0xAA,0x00,0x00,0x00,0x00,0x00,0x00,
#            0x00,0x00,0x00,0xAA]

# com.init(mode=2)
# 增加了底层驱动，可以直接通过cv2的0号设备读取摄像头
cap = cv2.VideoCapture(1)
cap.set(3, 1080)  # 设置摄像头输出宽
cap.set(4, 960)  # 设置摄像头输出高
# print("start reading video...")
# time.sleep(2.0)
# print("start working")
# 初始化右侧图像显示功能,优化了传输,
# 增加了imshow('name',frame)函数，与cv2的imshow保持一致
# 无需指定fps
# vd.init()

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=2,  # 限制角点数量
                      qualityLevel=0.5,  # 返回角点的质量（0.1-0.01，不超过1）
                      minDistance=7,  # 相邻角点的最小间距
                      blockSize=7)  # 计算角点时需要考虑的区域大小

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(11, 11),  # 用于计算局部干运动的窗口
                 maxLevel=5,  # 图像堆栈深度(图像金字塔)
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),  # 设置算何时退出搜索匹配
                 minEigThreshold=1e-4)  # 滤波点，除去点

# Create some random colors
color = np.random.randint(0, 255, (100, 3))


# 中位值平均滤波法
def MedianAverage(inputs, per):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    for tmp in inputs:
        tmp = np.delete(tmp, np.where(tmp == tmp.max())[0], axis=0)
        tmp = np.delete(tmp, np.where(tmp == tmp.min())[0], axis=0)
        mean.append(tmp.mean())
    return mean


# 限幅平均滤波算法  Amplitude：限制最大幅度
def AmplitudeLimitingAverage(inputs, per, Amplitude):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    tmpmean = inputs[0].mean()
    tmpnum = inputs[0][0]  # 上一次限幅后结果
    for tmp in inputs:
        for index, newtmp in enumerate(tmp):
            if np.abs(tmpnum - newtmp) > Amplitude:
                tmp[index] = tmpnum
            tmpnum = newtmp
        mean.append((tmpmean + tmp.mean()) / 2)
        tmpmean = tmp.mean()
    return mean


def Flow():
    track_len = 10
    detect_interval = 5
    tracks = []
    frame_idx = 0
    fps_counter = 0
    start_time = time.time()
    fps = 0
    filter_counter = 0
    while True:
        _ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame_gray = cv2.GaussianBlur(frame_gray,(7,7),5)
        # frame_gray = MedianAverage(frame_gray,30)
        vis = frame.copy()

        if len(tracks) > 0:
            img0, img1 = prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
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
            speed = np.zeros((cnt, 3))
            for i in range(0, len(tracks)):
                try:
                    speed[i][0] = tracks[i][1][0] - tracks[i][0][0]
                    speed[i][1] = tracks[i][1][1] - tracks[i][0][1]
                    speed[i][2] = math.sqrt(speed[i][0] * speed[i][0] + speed[i][1] * speed[i][1])
                except:
                    print('1')
            speed = speed[np.lexsort(-speed.T)]
            if (fps != None and len(tracks) != 0):
                if (len(tracks) > 3):
                    #                        print(len(self.tracks))
                    #                        print(len(speed))
                    for i in range(0, len(tracks) - 1):
                        try:
                            #                            print (speed[i][2] ,speed[i+1][2] )
                            if (speed[i][2] * 0.9 >= speed[i + 2][2]):
                                #                                    speed = np.delete(speed, 0, 0)
                                filter_counter = filter_counter + 1
                            else:
                                break
                        except:
                            print('2')
                    #                        speed = speed[speed[:,2] >= ( max(speed[:,2])) ]
                    speed = speed[speed[:, 2] <= (speed[filter_counter, 2])]
                    x = int(speed[0, 0] * fps * 10)
                    y = int(speed[0, 1] * fps * 10)
                    filter_counter = 0
                else:
                    #                        speed = speed[speed[:,2] == ( max(speed[:,2])) ]
                    x = int(np.mean(speed[:, 0]) * fps * 10)
                    y = int(np.mean(speed[:, 1]) * fps * 10)
                # Uart_buf = bytearray([0x55,0xAA,0x20, (x  & 0xff00) >>8 , x & 0x00ff ,(y & 0xff00) >>8  , y  & 0x00ff ,0 ,0 ,0 ,0 , 0xAA])
                # com.send(Uart_buf)
                print("speedX: %.1f ,speedY: %.1f" % (x, y))
            cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))

        if frame_idx % detect_interval == 0:  # 每5帧找一次特征角点
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    tracks.append([(x, y)])
                if (len(tracks) > 20):
                    tracks = np.delete(tracks, [0, 1], axis=0)  # 删除tracks的第0行和第1行

        frame_idx += 1
        prev_gray = frame_gray
        cv2.imshow("flow", vis)

        fps_counter = fps_counter + 1
        if (time.time() - start_time) > 1:
            fps = fps_counter / (time.time() - start_time)
            print("FPS: %.1f" % (fps))
            fps_counter = 0
            start_time = time.time()

        ch = cv2.waitKey(1)
        if ch == 27:
            break


def main():
    Flow()
    cv2.destroyAllWindows()


main()
