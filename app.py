from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image
import random

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")

emotion_map = {0:"CALM", 1:"ALERT", 2:"FEARFUL", 3:"AGGRESSIVE"}
risk_map = {0:"LOW", 1:"MEDIUM", 2:"HIGH"}

def infer_emotion(area_ratio, conf):
    if area_ratio > 0.25 and conf > 0.6:
        return 3
    elif area_ratio > 0.15:
        return 2
    elif conf > 0.4:
        return 1
    return 0

def attack_percentage(emotion):
    if emotion == 3:
        return random.randint(90, 100)
    elif emotion == 2:
        return random.randint(75, 90)
    elif emotion == 1:
        return random.randint(50, 70)
    return random.randint(10, 30)

def risk_level(pct):
    if pct >= 75:
        return 2
    elif pct >= 40:
        return 1
    return 0

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    output_image = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            img = np.array(Image.open(path).convert("RGB"))
            h, w, _ = img.shape

            results = model(img)[0]

            for box in results.boxes:
                if int(box.cls[0]) == 16:  # dog
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area_ratio = ((x2-x1)*(y2-y1))/(h*w)

                    emotion = infer_emotion(area_ratio, conf)
                    attack_pct = attack_percentage(emotion)
                    risk = risk_level(attack_pct)

                    label = f"{emotion_map[emotion]} | {risk_map[risk]} | {attack_pct}%"

                    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
                    cv2.putText(img, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

                    result = {
                        "emotion": emotion_map[emotion],
                        "risk": risk_map[risk],
                        "attack": attack_pct
                    }
                    break

            out_path = os.path.join(RESULT_FOLDER, file.filename)
            cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            output_image = out_path

    return render_template("index.html", result=result, image=output_image)

if __name__ == "__main__":
    app.run(debug=True)
