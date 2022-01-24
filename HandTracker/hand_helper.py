import cv2 as cv
import mediapipe as mp
import numpy as np


class HandHelper:
    def __init__(self):
        super().__init__()

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands()

        self.current_position = (0, 0)
        self.last_position = (0, 0)

        self.first_appearance = True

    def draw_pointer_line(self, frame, canvas):
        results = self.hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                for num, lm in enumerate(hand_lm.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if num == 8:
                        if self.first_appearance:
                            self.last_position = (cx, cy)
                            self.first_appearance = False
                        else:
                            self.last_position = self.current_position

                        self.current_position = (cx, cy)
                        cv.line(canvas, self.last_position, self.current_position, (0, 255, 0, 1), 5)

                # mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        frame = cv.cvtColor(frame, cv.COLOR_RGB2RGBA)
        frame = cv.add(frame, canvas)

        return frame
