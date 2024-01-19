import cv2
import mediapipe as mp
from math import hypot
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from google.protobuf.json_format import MessageToDict
import screen_brightness_control as sbc

# Left Hand for Brightness
# Right Hand for Volume

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.80)
mpDraw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volMin, volMax = volume.GetVolumeRange()[:2]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 2)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    left_lmList, right_lmList = [], []
    if results.multi_hand_landmarks and results.multi_handedness:
        for i in results.multi_handedness:
            label = MessageToDict(i)['classification'][0]['label']
            if label == 'Left':
                for lm in results.multi_hand_landmarks[0].landmark:
                    h, w, _ = img.shape
                    left_lmList.append([int(lm.x * w), int(lm.y * h)])
                mpDraw.draw_landmarks(img, results.multi_hand_landmarks[0], mpHands.HAND_CONNECTIONS)
            if label == 'Right':
                index = 0
                if len(results.multi_hand_landmarks) == 2:
                    index = 1
                for lm in results.multi_hand_landmarks[index].landmark:
                    h, w, _ = img.shape
                    right_lmList.append([int(lm.x * w), int(lm.y * h)])
                    mpDraw.draw_landmarks(img, results.multi_hand_landmarks[index], mpHands.HAND_CONNECTIONS)

    if left_lmList != []:
        x1, y1 = left_lmList[4][0], left_lmList[4][1]
        x2, y2 = left_lmList[8][0], left_lmList[8][1]

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        length = hypot(x2 - x1, y2 - y1)

        bright = np.interp(length, [15, 200], [0, 100])
        print("Brightness:", bright, "Length:", length)
        sbc.set_brightness(int(bright))

        # Display brightness value on the image
        cv2.putText(img, f'Brightness: {int(bright)}%', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    if right_lmList != []:
        x1, y1 = right_lmList[4][0], right_lmList[4][1]
        x2, y2 = right_lmList[8][0], right_lmList[8][1]

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        length = hypot(x2 - x1, y2 - y1)

        vol = np.interp(length, [15, 200], [volMin, volMax])
        vol_percentage = int(abs(vol))  # Take the absolute value for positive percentage
        print("Volume:", vol_percentage, "Length:", length)
        volume.SetMasterVolumeLevel(vol, None)

        # Display volume value on the image
        #cv2.putText(img, f'Volume: {vol_percentage}%', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,qcv2.LINE_AA)

    # Check for hand gesture to exit (making a fist with the left hand)
    if len(left_lmList) >= 9 and left_lmList[4][1] < left_lmList[8][1] and left_lmList[5][1] < left_lmList[9][1]:
        print("Exit gesture detected. Exiting...")
        break

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
