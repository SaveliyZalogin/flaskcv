import cv2 as cv
from flask import Flask, render_template, Response

app = Flask(__name__)


def get_stream():
    face = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

    mask = cv.imread('images/steve_face.jpg')
    cap = cv.VideoCapture(cv.CAP_V4L2)

    if not cap.isOpened():
        print('Error')
        exit()
    while True:
        resp, frame = cap.read()
        frame = cv.flip(frame, 1)

        if not resp:
            print('Something went wrong :(')
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(frame_gray, 1.1, 19)

        for x, y, w, h in faces:
            mask = cv.resize(mask, (w, h))
            frame[y:y + h, x:x + w] = mask

        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(get_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

