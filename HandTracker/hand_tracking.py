import cv2 as cv
import mediapipe as mp
import numpy as np
from hand_helper import HandHelper

cap = cv.VideoCapture(0)

_, frame = cap.read()
canvas = np.zeros((frame.shape[0], frame.shape[1], 4), np.uint8)

hand_helper = HandHelper()

first_appearance = True

while True:
    resp, frame = cap.read()
    frame = cv.flip(frame, 1)

    if not resp:
        print('Something went wrong :(')
        break

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    result = cv.cvtColor(hand_helper.draw_pointer_line(frame_rgb, canvas), cv.COLOR_BGR2RGBA)

    cv.imshow('frame', result)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

