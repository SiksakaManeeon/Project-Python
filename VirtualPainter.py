import cv2
import numpy as np
import os
import HandTrackingModule as htm

#######################
brushThickness = 20
eraserThickness = 100
#######################

folderPath = 'Header'
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header = overlayList[0]
#default brush color
drawColor = (0, 0, 255)
xp, yp = 0, 0
xp1, yp1 = 0, 0
cap = cv2.VideoCapture(0)
#dimension
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.9)

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    #import image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    #find hand landmarks
    img = detector.findHands(img)
    lmList0, handsType0 = detector.findPosition0(img, draw=True)
    lmList1, handsType1 = detector.findPosition1(img, draw=True)
    
    if len(lmList0) != 0:
        #tip of index, middle and ring fingers
        x1, y1 = lmList0[8][1:]
        x2, y2 = lmList0[12][1:]
        x3, y3 = lmList0[16][1:]
        #check which fingers are up
        fingers0 = detector.fingersUp0()
        
        #color select
        if fingers0[1] and fingers0[2] and fingers0[3]==False and fingers0[4]==False:
            xp, yp = 0, 0

            if y1 < 95:
                #red brush
                if 285 < x1 < 380:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                #blue brush
                elif 475 < x1 < 570:
                    header = overlayList[1]
                    drawColor = (255, 128, 0)
                #green brush
                elif 665 < x1 < 760:
                    header = overlayList[2]
                    drawColor = (51, 255, 153)
                #yellow brush
                elif 855 < x1 < 950:
                    header = overlayList[3]
                    drawColor = (51, 255, 255)
                #gray brush
                elif 1045 < x1 < 1140:
                    header = overlayList[4]
                    drawColor = (192, 192, 192)

            cv2.rectangle(img, (x1, y1-25), (x2,y2+25), drawColor, cv2.FILLED)

        #draw
        if fingers0[1] and fingers0[2]==False and fingers0[3]==False and fingers0[4]==False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    if len(lmList1) != 0:
        x11, y11 = lmList1[8][1:]
        fingers1 = detector.fingersUp1()

        if fingers1[1] and fingers1[2] and fingers1[3]:
            cv2.circle(img, (x11, y11), 15, (0, 0, 0), cv2.FILLED)
            if xp1 == 0 and yp1 == 0:
                xp1, yp1 = x11, y11
            cv2.line(img, (xp1, yp1), (x11, y11), (0, 0, 0), eraserThickness)
            cv2.line(imgCanvas, (xp1, yp1), (x11, y11), (0, 0, 0), eraserThickness)
            xp1, yp1 = x11, y11

    #merge layers
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv) #
    img = cv2.bitwise_or(img, imgCanvas) #

    img[0:95, 0:1280] = header
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow('Image', img)

    #cv2.imshow('Canvas', imgCanvas) #
    #cv2.imshow('Inv', imgInv) #
    cv2.waitKey(1)