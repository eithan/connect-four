import base64
import io
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "detect", "train", "weights", "best.pt")

model = YOLO(MODEL_PATH)

# Class indices in the model
CLASS_RED = 1
CLASS_YELLOW = 2
# CLASS_EMPTY = 0 — not drawn

# BGR colors for OpenCV drawing
COLOR_BOARD = (255, 80, 0)    # blue
COLOR_RED = (0, 0, 220)       # red
COLOR_YELLOW = (0, 220, 220)  # yellow
BOX_THICKNESS = 3
BOARD_THICKNESS = 4
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 2


def run_inference(image_bytes: bytes) -> tuple[str, dict]:
    """Run YOLO inference and return annotated image as base64 + stats."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")

    results = model.predict(img_bgr, imgsz=640, verbose=False)[0]
    boxes = results.boxes

    red_count = 0
    yellow_count = 0
    board_coords = []  # collect all piece coords to infer board bounds

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])

        if cls == CLASS_RED:
            red_count += 1
            board_coords.append((x1, y1, x2, y2))
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), COLOR_RED, BOX_THICKNESS)
            label = f"Red {conf:.2f}"
            cv2.putText(img_bgr, label, (x1, max(y1 - 6, 12)),
                        FONT, FONT_SCALE, COLOR_RED, FONT_THICKNESS, cv2.LINE_AA)

        elif cls == CLASS_YELLOW:
            yellow_count += 1
            board_coords.append((x1, y1, x2, y2))
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), COLOR_YELLOW, BOX_THICKNESS)
            label = f"Yellow {conf:.2f}"
            cv2.putText(img_bgr, label, (x1, max(y1 - 6, 12)),
                        FONT, FONT_SCALE, COLOR_YELLOW, FONT_THICKNESS, cv2.LINE_AA)

    # Draw inferred board bounding box if any pieces were found
    if board_coords:
        bx1 = min(c[0] for c in board_coords)
        by1 = min(c[1] for c in board_coords)
        bx2 = max(c[2] for c in board_coords)
        by2 = max(c[3] for c in board_coords)
        # Add a small margin
        margin = 10
        bx1 = max(0, bx1 - margin)
        by1 = max(0, by1 - margin)
        bx2 = min(img_bgr.shape[1] - 1, bx2 + margin)
        by2 = min(img_bgr.shape[0] - 1, by2 + margin)
        cv2.rectangle(img_bgr, (bx1, by1), (bx2, by2), COLOR_BOARD, BOARD_THICKNESS)
        cv2.putText(img_bgr, "Board", (bx1, max(by1 - 8, 14)),
                    FONT, FONT_SCALE + 0.1, COLOR_BOARD, FONT_THICKNESS, cv2.LINE_AA)

    # Encode result as base64 PNG
    success, buf = cv2.imencode(".png", img_bgr)
    if not success:
        raise ValueError("Could not encode result image")

    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    stats = {
        "red": red_count,
        "yellow": yellow_count,
        "board_detected": len(board_coords) > 0,
    }
    return b64, stats


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    image_bytes = file.read()
    try:
        b64_img, stats = run_inference(image_bytes)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

    return jsonify({"image": b64_img, "stats": stats})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
