import numpy as np
import imutils
import cv2
import cv2.aruco as aruco
import multiprocessing
#import V_Display as vd

mtx = np.array([
    [235.30964269, 0, 110.68578335],
    [0, 229.75119518, 107.7683592],
    [0, 0, 1],
])
dist = np.array([-0.57063508, 0.8469956, -0.00954837, 0.00477835, -0.47724227])

def Aruco(frameQueue,arucoX,arucoY):
    while(True):
        if not frameQueue.empty():
            frame = frameQueue.get()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
            parameters = aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if np.all(ids != None):
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[0], 0.05, mtx, dist)
                aruco.drawDetectedMarkers(frame, corners, ids)
                if ids[0][0] == 12:
                    aruco.drawAxis(frame, mtx, dist, rvec[0], tvec[0], 0.1)
                numbers = ids.shape[0]
                send = np.zeros([numbers, 3], dtype=float)
                for t in range(numbers):
                    a = corners[t]
                    x1 = a[0][1][0]
                    y1 = a[0][1][1]
                    x2 = a[0][2][0]
                    y2 = a[0][2][1]
                    x3 = a[0][3][0]
                    y3 = a[0][3][1]
                    x4 = a[0][0][0]
                    y4 = a[0][0][1]
                    try:
                        k1 = (y3 - y1) / (x3 - x1)
                        b1 = (x3 * y1 - x1 * y3) / (x3 - x1)
                        k2 = (y4 - y2) / (x4 - x2)
                        b2 = (x4 * y2 - x2 * y4) / (x4 - x2)
                        x5 = -(b1 - b2) / (k1 - k2)
                        y5 = k1 * x5 + b1
                        send[t][0] = ids[t][0]
                        send[t][1] = int(x5)
                        send[t][2] = int(y5)
                        ARUCOX = send[t][1]
                        ARUCOY = send[t][2]
                    except:
                        continue
                cv2.putText(frame, "Id: " + str(ids), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                ARUCOY = 0
                ARUCOX = 0
            try:
                arucoX.put(ARUCOX, False)
                arucoY.put(ARUCOY, False)
            except:
                continue
            cv2.imshow("Aruco",frame)
            #vd.show(frame)
        else:
            continue

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)
    arucoX = multiprocessing.Queue(maxsize=1)
    arucoY = multiprocessing.Queue(maxsize=1)
    frameQueue = multiprocessing.Queue(maxsize=1)
    processAruco = multiprocessing.Process(target=Aruco, args=(frameQueue, arucoX, arucoY))
    processAruco.start()
    aruco_X = 0
    aruco_Y = 0
    HoughX = 160
    HoughX = 120
    while(True):
        ret, frame = cap.read()
        try:
            frameQueue.put(frame, False)
        except:
            continue
        frame = imutils.resize(frame, width=160)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=30, minRadius=0, maxRadius=0)
        if circles is not None:
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(gray, (i[0], i[1]), i[2], (255, 0, 0), 2)
                # draw the center of the circle
                cv2.circle(gray, (i[0], i[1]), 2, (255, 0, 0), 3)
                HoughX = int(i[0])
                HoughY = int(i[1])
        else:
            HoughX = 160
            HoughX = 120
        #print("Circles X:", HoughX, "Circles Y:", HoughY)
        #cv2.imshow("Hough", frame)
        vd.show(frame)
        if not arucoX.empty():
            aruco_X = arucoX.get()
            aruco_Y = arucoY.get()
        #print("aruco-x: %d ,aruco-y: %d" % (aruco_X, aruco_Y))
        print(HoughX,HoughY,aruco_X,aruco_Y)
