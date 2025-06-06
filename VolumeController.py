import math
import cv2
import numpy as np
import time
from HandDetection.HandDetectionModule import HandDetectionModule
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

volume_range = volume.GetVolumeRange()
minimum_volume, maximum_volume = volume_range[0], volume_range[1]


capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)

detect = HandDetectionModule(detectionCon=0.7)
cTime = 0
pTime = 0

vol = 0
volumeBar = 400

while True:
    ret, image = capture.read()
    if not ret:
        continue

    image = detect.handFinder(image)
    lmList = detect.positionFinder(image, draw=False)
    if len(lmList) != 0:

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2


        cv2.circle(image, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.circle(image, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        print(length)

        '''
            myVolume range = -65.0 to 0
            hand range = 20 to 160
        '''

        vol = np.interp(length, [20, 160], [minimum_volume, maximum_volume])
        volumeBar = np.interp(length, [20, 160], [400, 150])
        percentage = np.interp(length, [20, 160], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)

        if length < 20:
            cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 255), 3)
        cv2.rectangle(image, (50, int(volumeBar)), (85, 400), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, f'{int(percentage)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime


    cv2.putText(image, f'FPS : {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow("Webcam with hand-Detection", image)

    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()