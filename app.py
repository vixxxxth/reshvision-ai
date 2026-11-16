
import base64
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from fruit_ai import analyze_produce

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


@app.route("/")
def home():
    return render_template("index.html")


def decode_image(data_url):
    if not data_url:
        raise ValueError("Please capture an image or choose a file.")

    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    try:
        raw = base64.b64decode(data_url, validate=True)
    except Exception as exc:
        raise ValueError("Could not decode the uploaded image.") from exc

    image = cv2.imdecode(
        np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Unsupported or invalid image.")
    return image


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(silent=True) or {}
        image = decode_image(data.get("image", ""))

        try:
            cv2.imwrite("captured_produce.jpg", image)
        except Exception:
            pass

        h, w = image.shape[:2]
        resized = cv2.resize(image, (400, 400))
        b, g, r = cv2.split(resized)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)

        result = analyze_produce(image)
        result["image_dimensions"] = {
            "width": int(w),
            "height": int(h)
        }
        result["features"] = {
            "red": round(float(np.mean(r)), 1),
            "green": round(float(np.mean(g)), 1),
            "blue": round(float(np.mean(b)), 1),
            "hue": round(float(np.mean(hsv[:, :, 0])), 1),
            "saturation": round(float(np.mean(hsv[:, :, 1])), 1),
            "brightness": round(float(np.mean(hsv[:, :, 2])), 1),
            "texture": round(float(np.std(gray)), 1),
            "edge_density": round(
                float(np.count_nonzero(edges) / edges.size * 100), 1
            ),
            "dark_area": round(
                float(np.count_nonzero(gray < 65) / gray.size * 100), 1
            )
        }

        return jsonify({
            "status": "success",
            "recognition": result,
            "message": "Image analysis completed."
        })

    except ValueError as exc:
        return jsonify({
            "status": "error",
            "error": str(exc)
        }), 400

    except Exception:
        app.logger.exception("Image analysis failed")
        return jsonify({
            "status": "error",
            "error": "Analysis failed. Try a clear, well-lit image."
        }), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "app": "FreshVision AI"
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False
    )
