import os

import cv2
from flask import Flask, Response, render_template

from hand_tracking import HandTracker

app = Flask(__name__)

camera_index = int(os.getenv("CAMERA_INDEX", "0"))
tracker = HandTracker(camera_index=camera_index)
tracker.start()


@app.route("/")
def index():
    return render_template("stream.html")


def generate():
    while True:
        frame = tracker.get_frame()
        if frame is None:
            continue

        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            encoded.tobytes() +
            b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8000")),
        debug=True,
        threaded=True,
        use_reloader=False,
    )
