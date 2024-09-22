from ultralytics import YOLO
from flask import request, Response, Flask, send_from_directory
from waitress import serve
from PIL import Image
import json

app = Flask(__name__)

@app.route("/")
def root():
    with open("index.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"])
def detect():
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(Image.open(buf.stream))
    return Response(
      json.dumps(boxes),  
      mimetype='application/json'
    )

@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')


def detect_objects_on_image(buf):
    
    model = YOLO("best.pt")
    results = model.predict(buf)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
          round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
          x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output

serve(app, host='0.0.0.0', port=8080)