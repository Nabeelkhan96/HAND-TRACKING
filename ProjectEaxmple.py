import cv2
import numpy as np
import HandtrackingModule as htm
import autopy

##########################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7
#########################
drawColor = (255, 0, 255)
brushThickness = 10
eraserThickness = 100
xp, yp = 0, 0
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
imgCanvas = np.zeros((hCam, wCam, 3), np.uint8)
while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x0, y0 = lmList[4] [1:]
        x1, y1 = lmList [8][1:]
        x2, y2 = lmList [12][1:]
        x3, y3 = lmList [16][1:]
        x4, y4 = lmList[20][1:]
        #print(x1, y1, x2, y2)

        fingers = detector.fingersUp()
        print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                     (255, 0, 255), 2)

        if fingers [1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 8. Both Index and middle fingers are up : Clicking Mode
        if fingers [1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            # 10. Click mouse if distance short
            if length < 20:
               cv2.circle(img, (lineInfo [4], lineInfo[5]),15, (0, 255, 0), cv2.FILLED)
               autopy.mouse.click()

        # Draw funtion
        if fingers[3] == 1 and fingers[4] == 1:
            cv2.circle(img, (x3, y3), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x3, y0
                length, img, lineInfo = detector.findDistance(4, 8, img)
                print(length)

                if length < 20:

                 cv2.line(img, (xp, yp), (x0, y0), drawColor, brushThickness)
                 cv2.line(imgCanvas, (xp, yp), (x0, y0), drawColor, brushThickness)

            # if drawColor == (0, 0, 0):
            #     cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
            #     cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            #
            # else:
            #     cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            #     cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

                xp, yp = x0, y0
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    cv2.imshow("Image", img)
    #cv2.imshow("Canvas", imgCanvas)
    #cv2.imshow("Inv", imgInv)

    cv2.waitKey(1)