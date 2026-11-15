import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

MODEL_PATH = os.path.expanduser(
    "~/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth"
)

weights = MobileNet_V3_Small_Weights.DEFAULT
categories = weights.meta["categories"]
preprocess = weights.transforms()
model = None
model_status = "Local model unavailable"

try:
    if os.path.isfile(MODEL_PATH):
        model = mobilenet_v3_small(weights=None)
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )
        model.eval()
        model_status = "Loaded offline"
except Exception as exc:
    model_status = f"Model unavailable: {type(exc).__name__}"
    model = None

PRODUCE = {
    "avocado": "Avocado",
    "banana": "Banana",
    "mango": "Mango",
    "apple": "Apple",
    "tomato": "Tomato",
    "papaya": "Papaya",
    "guava": "Guava",
    "lemon": "Lemon",
    "orange": "Orange",
    "strawberry": "Strawberry",
    "pineapple": "Pineapple",
    "watermelon": "Watermelon",
    "cucumber": "Cucumber",
    "brinjal": "Brinjal / Eggplant",
    "potato": "Potato",
    "onion": "Onion",
    "bell_pepper": "Bell Pepper",
    "other": "Other Produce",
}

# Classes the existing ImageNet model may recognize.
CLASS_MAP = {
    "banana": "banana",
    "granny smith": "apple",
    "red apple": "apple",
    "orange": "orange",
    "lemon": "lemon",
    "pineapple": "pineapple",
    "strawberry": "strawberry",
    "fig": "other",
    "jackfruit": "other",
    "custard apple": "other",
    "pomegranate": "other",
    "bell pepper": "bell_pepper",
    "cucumber": "cucumber",
    "zucchini": "cucumber",
    "potato": "potato",
    "broccoli": "other",
    "cauliflower": "other",
    "head cabbage": "other",
    "tomato": "tomato",
}


def get_colors(image):
    if image is None or image.size == 0:
        raise ValueError("Empty image")

    h, w = image.shape[:2]
    crop = image[int(h * .2):int(h * .8), int(w * .2):int(w * .8)]
    if crop.size == 0:
        crop = image

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    total = max(hue.size, 1)

    def pct(mask):
        return round(100 * np.count_nonzero(mask) / total, 1)

    return {
        "green": pct((hue >= 35) & (hue <= 85) & (sat >= 35)),
        "yellow": pct((hue >= 20) & (hue < 35) & (sat >= 35) & (val >= 50)),
        "orange": pct((hue >= 8) & (hue < 22) & (sat >= 45) & (val >= 45)),
        "red": pct((((hue < 8) | (hue >= 170)) & (sat >= 45) & (val >= 40))),
        "brown": pct((hue >= 5) & (hue <= 25) & (sat >= 35) & (val < 125)),
        "dark": pct(val < 65),
        "brightness": round(float(np.mean(val)), 1),
    }


def detect_produce(image):
    if model is None:
        return None, None, []

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0)

    with torch.no_grad():
        probs = model(tensor).softmax(dim=1)[0]

    values, indices = torch.topk(probs, 5)
    top = [
        {
            "label": categories[int(i)],
            "score": round(float(v) * 100, 2),
        }
        for v, i in zip(values, indices)
    ]

    for item in top:
        label = item["label"].lower()
        for keyword, produce in CLASS_MAP.items():
            if keyword in label and item["score"] >= 15:
                return produce, item["score"], top

    return None, None, top


def estimate_ripeness(kind, c):
    g, y, o = c["green"], c["yellow"], c["orange"]
    r, b, d = c["red"], c["brown"], c["dark"]

    if kind == "banana":
        if b >= 18 or d >= 35:
            return "Possibly overripe", "0 days — already ripe or past ripe"
        if y >= 12 and g < 10:
            return "Likely ripe", "0 days — likely already ripe"
        if y >= 8 and g >= 5:
            return "Ripening", "Approximately 1–3 days"
        if g >= 12:
            return "Likely unripe", "Approximately 2–5 days"
        return "Uncertain", "Cannot estimate reliably from color"

    if kind == "avocado":
        g = float(c.get("green", 0))
        r = float(c.get("red", 0))
        y = float(c.get("yellow", 0))
        o = float(c.get("orange", 0))
        d = float(c.get("dark", 0))
        b = float(c.get("brown", 0))
        warm = r + y + o
    
        # Color is only a rough clue; avocado varieties differ.
        if d >= 35 or b >= 30:
            return (
                "Estimated ripeness: approximately 85% (possible overripe/damaged; low confidence)",
                "No reliable days estimate; check firmness and inspect before eating"
            )
        elif g >= 35 and warm < 20 and d < 25 and b < 25:
            return (
                "Estimated ripeness: approximately 20% (possibly unripe; low confidence)",
                "Approx. 3–6 days to ripen if firm; low-confidence estimate, check daily"
            )
        elif g >= 20 and warm < 25 and d < 30 and b < 25:
            return (
                "Estimated ripeness: approximately 55% (possibly approaching ripe; low confidence)",
                "Approx. 1–3 days to ripen if still firm; low-confidence estimate, check daily"
            )
        else:
            return (
                "Estimated ripeness: approximately 45% (uncertain; low confidence)",
                "Approx. 2–4 days as a rough guide only; color is inconclusive, check firmness daily"
            )

    if kind == "apple":
        if b >= 18 or d >= 35:
            return (
                "Dark or brown areas detected; inspect the apple",
                "Not reliably predictable from color"
            )
        if r + y + o >= 18:
            return (
                "Red/yellow color detected; eating readiness unconfirmed",
                "Days to ripe cannot be reliably estimated for apples"
            )
        return (
            "Apple selected; color alone is inconclusive",
            "Days to ripe cannot be reliably estimated for apples"
        )

    if kind in ("orange", "lemon", "pineapple", "guava"):
        return "Color measured; variety matters", "Days to ripe unavailable reliably"

    if kind in ("watermelon", "cucumber", "brinjal", "potato",
                "onion", "bell_pepper"):
        return "Surface color measured", "Days to ripe unavailable reliably"

    return "Produce not confidently identified", "Unable to estimate"


def recognize_fruit(image_bgr, produce_type="avocado"):
    selected = str(produce_type or "avocado").lower().strip()
    if selected not in PRODUCE:
        selected = "other"

    colors = get_colors(image_bgr)
    detected, score, top = detect_produce(image_bgr)

    # The user-selected produce is authoritative.
    # The general-purpose model is not reliable enough to override it.
    kind = selected
    source = "User selection; offline color analysis"

    ripeness, days = estimate_ripeness(kind, colors)

    return {
        "predicted_label": PRODUCE.get(kind, "Other Produce"),
        "selected_produce": PRODUCE.get(selected, "Other Produce"),
        "detection_source": source,
        "model_score_percent": None,
        "ripeness": ripeness,
        "days_to_ripe": days,
        "method": "Offline image classification + approximate color rules",
        "warning": (
            "Experimental estimate only. This model is not trained for "
            "ripeness or days-to-ripen prediction. Results depend on "
            "lighting, variety, temperature and storage conditions."
        ),
        "colors": colors,
        "model_status": model_status,
        "model_is_trained_for_ripeness": False,
        "top_predictions": top,
    }
