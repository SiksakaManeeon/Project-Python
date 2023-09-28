import cv2
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        #for Hands method
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        #call hand detection from mediapipe
        self.mpHands = mp.solutions.hands 
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        #spot the dot if detecting hand
        self.mpDraw = mp.solutions.drawing_utils
        #tip of fingers
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        #change image from BGR to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #return hand landmarks
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
    
    def findPosition0(self, img, handNo=-1, draw=True):
        self.lmList0 = []
        handsType0 = ''
        if self.results.multi_hand_landmarks:
            handsType0 = self.results.multi_handedness[0].classification[0].label
            myHand = self.results.multi_hand_landmarks[handNo]
            for idlm, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    self.lmList0.append([idlm, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 8, (0, 0, 128), cv2.FILLED)

        return self.lmList0, handsType0
    
    def findPosition1(self, img, handNo=-2, draw=True):
        self.lmList1 = []
        handsType1 = ''
        try:
            if self.results.multi_hand_landmarks:
                handsType1 = self.results.multi_handedness[1].classification[0].label
                myHand1 = self.results.multi_hand_landmarks[handNo]
                for idlm1, lm1 in enumerate(myHand1.landmark):
                        h1, w1, c1 = img.shape
                        cx1, cy1 = int(lm1.x*w1), int(lm1.y*h1)
                        self.lmList1.append([idlm1, cx1, cy1])
                        if draw:
                            cv2.circle(img, (cx1, cy1), 8, (0, 255, 128), cv2.FILLED)
        except:
            pass

        return self.lmList1, handsType1

    def fingersUp0(self):
        fingers0 = []
        
        #thumb
        if self.lmList0[self.tipIds[0]][1] > self.lmList0[self.tipIds[0]-1][1]:
            fingers0.append(1)
        else:
            fingers0.append(0)

        #4 fingers
        for id in range(1, 5):
            if self.lmList0[self.tipIds[id]][2] < self.lmList0[self.tipIds[id]-2][2]:
                fingers0.append(1)
            else:
                fingers0.append(0)
        
        return fingers0
    
    def fingersUp1(self):
        fingers1 = []

        #thumb
        if self.lmList1[self.tipIds[0]][1] < self.lmList1[self.tipIds[0]-1][1]:
            fingers1.append(1)
        else:
            fingers1.append(0)

        #4 fingers
        for id in range(1, 5):
            if self.lmList1[self.tipIds[id]][2] < self.lmList1[self.tipIds[id]-2][2]:
                fingers1.append(1)
            else:
                fingers1.append(0)
        
        return fingers1
    
def main():
    #select camera
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        #get image in BGR
        success, img = cap.read()
        img = cv2.flip(img, 1)
        #find hands
        img = detector.findHands(img)
        #after find hands spot positions
        lmList0, handsType0 = detector.findPosition0(img)
        lmList1, handsType1 = detector.findPosition1(img)
        print(handsType0, handsType1)
        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()