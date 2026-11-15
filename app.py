from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from fruit_ai import recognize_fruit
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.json
    image_data = data["image"]

    # Decode browser camera image
    encoded = image_data.split(",")[1]
    image_bytes = base64.b64decode(encoded)
    np_array = np.frombuffer(image_bytes, np.uint8)

    image = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return jsonify({
            "error": "Unable to process image"
        }), 400

    # ---------------------------------
    # SAVE IMAGE FOR AI MODEL
    # ---------------------------------

    cv2.imwrite(
        "captured_produce.jpg",
        image
    )

    # ---------------------------------
    # IMAGE PROCESSING
    # ---------------------------------

    processed = cv2.resize(
        image,
        (500, 500)
    )

    # RGB FEATURES
    b, g, r = cv2.split(processed)

    red = float(np.mean(r))
    green = float(np.mean(g))
    blue = float(np.mean(b))

    # HSV FEATURES
    hsv = cv2.cvtColor(
        processed,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(hsv)

    hue = float(np.mean(h))
    saturation = float(np.mean(s))
    brightness = float(np.mean(v))

    # GRAYSCALE
    gray = cv2.cvtColor(
        processed,
        cv2.COLOR_BGR2GRAY
    )

    # TEXTURE
    texture = float(
        np.std(gray)
    )

    # EDGE DETECTION
    edges = cv2.Canny(
        gray,
        80,
        160
    )

    edge_density = (
        np.count_nonzero(edges)
        / edges.size
    ) * 100

    # DARK REGION ANALYSIS
    lower_dark = np.array(
        [0, 0, 0]
    )

    upper_dark = np.array(
        [180, 255, 70]
    )

    dark_mask = cv2.inRange(
        hsv,
        lower_dark,
        upper_dark
    )

    dark_percentage = (
        np.count_nonzero(dark_mask)
        / dark_mask.size
    ) * 100

    # ---------------------------------
    # RESPONSE
    # ---------------------------------

    return jsonify({

        "status": "success",
        "recognition": recognize_fruit(
         image,
    data.get("produce", "avocado")
), 
        "features": {

            "red": round(red, 1),
            "green": round(green, 1),
            "blue": round(blue, 1),

            "hue": round(hue, 1),
            "saturation": round(
                saturation, 1
            ),
            "brightness": round(
                brightness, 1
            ),

            "texture": round(
                texture, 1
            ),

            "edge_density": round(
                edge_density, 1
            ),

            "dark_area": round(
                dark_percentage, 1
            )
        },

        "message":
            "FreshVision analysis completed."
    })


if __name__ == "__main__":

    print()
    print("=" * 50)
    print("FreshVision AI")
    print("Camera + Image Processing System")
    print("=" * 50)
    print()
    print("Open: http://127.0.0.1:5000")
    print()

    app.run(
        debug=True,
        port=5000
    )
