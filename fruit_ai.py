
import warnings

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    from torchvision.models import (
        mobilenet_v3_small,
        MobileNet_V3_Small_Weights,
    )
except Exception:
    torch = None
    mobilenet_v3_small = None
    MobileNet_V3_Small_Weights = None


_model = None
_weights = None
_categories = []
_preprocess = None
_model_status = "Model not loaded"

LABEL_TO_PRODUCE = {
    "banana": "Banana",
    "Granny Smith": "Apple",
    "orange": "Orange",
    "lemon": "Lemon",
    "pineapple": "Pineapple",
    "strawberry": "Strawberry",
    "bell pepper": "Bell pepper",
    "cucumber": "Cucumber",
    "zucchini": "Zucchini",
    "broccoli": "Broccoli",
    "cauliflower": "Cauliflower",
    "head cabbage": "Cabbage",
    "artichoke": "Artichoke",
    "corn": "Corn",
    "mushroom": "Mushroom",
    "acorn squash": "Squash",
    "fig": "Fig",
    "pomegranate": "Pomegranate",
}


def load_model():
    global _model, _weights, _categories
    global _preprocess, _model_status

    if _model is not None:
        return

    if torch is None:
        _model_status = "PyTorch is unavailable"
        return

    try:
        _weights = MobileNet_V3_Small_Weights.DEFAULT
        _categories = _weights.meta["categories"]
        _preprocess = _weights.transforms()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _model = mobilenet_v3_small(weights=_weights)

        _model.eval()
        _model_status = "ImageNet MobileNetV3 Small loaded"

    except Exception as exc:
        _model = None
        _model_status = "Model unavailable: " + type(exc).__name__


def detect_produce(image_bgr):
    load_model()

    if _model is None:
        return {
            "label": None,
            "score_percent": None,
            "top_predictions": [],
            "status": _model_status,
        }

    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = _preprocess(Image.fromarray(rgb)).unsqueeze(0)

        with torch.inference_mode():
            probabilities = _model(tensor).softmax(dim=1)[0]

        values, indices = torch.topk(probabilities, 5)

        top = [
            {
                "label": _categories[int(index)],
                "score_percent": round(float(value) * 100, 2),
            }
            for value, index in zip(values, indices)
        ]

        first_label = top[0]["label"]
        produce = None

        for label, name in LABEL_TO_PRODUCE.items():
            if label.lower() == first_label.lower():
                produce = name
                break

        return {
            "label": produce,
            "score_percent": top[0]["score_percent"],
            "top_predictions": top,
            "status": _model_status,
        }

    except Exception as exc:
        return {
            "label": None,
            "score_percent": None,
            "top_predictions": [],
            "status": "Inference error: " + type(exc).__name__,
        }


def get_color_features(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    total = max(hue.size, 1)

    def percentage(mask):
        return round(
            100 * float(np.count_nonzero(mask)) / total, 1
        )

    return {
        "green": percentage(
            (hue >= 35) & (hue <= 85) & (saturation >= 35)
        ),
        "yellow": percentage(
            (hue >= 20) & (hue < 35)
            & (saturation >= 35) & (value >= 50)
        ),
        "brown": percentage(
            (hue >= 5) & (hue <= 25)
            & (saturation >= 25) & (value < 125)
        ),
        "dark": percentage(value < 65),
        "brightness": round(float(np.mean(value)), 1),
    }


def assess_condition(kind, colors):
    if not kind:
        ripeness = "Unknown: produce was not confidently identified"
    elif kind.lower() == "banana":
        if colors["brown"] >= 12:
            ripeness = "Brown areas detected; inspect the banana closely"
        elif colors["yellow"] >= 15 and colors["green"] < 10:
            ripeness = "Yellow color detected; check firmness to assess ripeness"
        elif colors["green"] >= 12:
            ripeness = "Green color detected; it may be less ripe"
        else:
            ripeness = "Ripeness is uncertain from color alone"
    else:
        ripeness = (
            "Visual color recorded; ripeness cannot be confirmed "
            "from this image alone"
        )

    discoloration = colors["dark"] + colors["brown"]

    if discoloration >= 30:
        freshness = (
            "Dark or brown areas detected; inspect for damage or decay"
        )
    elif discoloration >= 15:
        freshness = (
            "Some discoloration detected; check smell and texture"
        )
    else:
        freshness = (
            "No strong dark-color signal detected; freshness is "
            "not guaranteed"
        )

    return ripeness, freshness


def storage_advice(kind):
    advice = {
        "banana": "Keep green bananas at room temperature. Refrigerate ripe bananas to slow further ripening.",
        "apple": "Refrigerate apples for longer storage.",
        "orange": "Keep at room temperature briefly or refrigerate for longer storage.",
        "lemon": "Refrigerate for longer storage.",
        "avocado": "Ripen firm avocados at room temperature; refrigerate when ripe.",
        "strawberry": "Refrigerate dry and unwashed; wash just before eating.",
        "pineapple": "Refrigerate cut pineapple promptly.",
        "cucumber": "Refrigerate and keep dry.",
        "bell pepper": "Refrigerate in the crisper drawer.",
        "broccoli": "Refrigerate promptly in a loose or perforated bag.",
        "potato": "Store in a cool, dark, ventilated place.",
        "mango": "Ripen at room temperature; refrigerate when ripe.",
    }

    return advice.get(
        (kind or "").lower(),
        "Use produce-specific storage guidance. Refrigerate cut produce promptly.",
    )


def analyze_produce(image_bgr):
    colors = get_color_features(image_bgr)
    detection = detect_produce(image_bgr)

    label = detection["label"]
    ripeness, freshness = assess_condition(label, colors)

    return {
        "predicted_label": label or "Not confidently identified",
        "detection_source": (
            "Pretrained ImageNet MobileNetV3 Small"
            if label else "No supported produce label matched"
        ),
        "model_score_percent": detection["score_percent"],
        "confidence_note": (
            "Raw ImageNet class score, not calibrated confidence "
            "or ripeness confidence."
        ),
        "ripeness": ripeness,
        "freshness": freshness,
        "days_to_ripe": (
            "Not available: no validated ripening-time model is installed."
        ),
        "storage_recommendation": storage_advice(label),
        "colors": colors,
        "model_status": detection["status"],
        "model_is_trained_for_ripeness": False,
        "top_predictions": detection["top_predictions"],
        "warning": (
            "Experimental visual aid only. This model is not trained "
            "to assess produce freshness or ripeness. Lighting, variety "
            "and background can affect results. Inspect produce yourself."
        ),
    }
