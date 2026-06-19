"""
Face-shape detection from a single photo.

Pipeline: OpenCV (image handling + drawing) + dlib (HOG face detector + 68-point
facial-landmark predictor) -> geometric measurements -> rule-based classifier.

Improvements over the original StyleZone script:
  * No fixed 500x500 resize (that squished the aspect ratio and corrupted the
    length-vs-width proportions the whole thing depends on). We keep the native
    aspect ratio and only downscale very large images.
  * Forehead width is taken from the outer-eyebrow landmarks (17<->26) instead of
    the fragile KMeans hair/skin segmentation, so it doesn't break on hairlines,
    lighting, or bald/light hair.
  * Classification uses NORMALISED RATIOS (length/width, region-to-region width
    ratios, jaw angle) instead of absolute pixel thresholds, so it generalises
    across image sizes and framing.
  * Covers 8 shapes (adds Oval and Heart, which the original was missing) instead
    of defaulting almost everyone to "oblong".
"""
from __future__ import annotations

import bz2
import math
import os
import urllib.request

import cv2
import dlib
import numpy as np

# --------------------------------------------------------------------------- #
# Model loading (downloads the standard dlib 68-landmark model on first run)
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_HERE, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
MODEL_URL = (
    "https://github.com/davisking/dlib-models/raw/master/"
    "shape_predictor_68_face_landmarks.dat.bz2"
)

_detector = None
_predictor = None


def _ensure_model() -> None:
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    bz2_path = MODEL_PATH + ".bz2"
    print("Downloading dlib landmark model (~64 MB, one time)...")
    urllib.request.urlretrieve(MODEL_URL, bz2_path)
    with bz2.open(bz2_path, "rb") as src, open(MODEL_PATH, "wb") as dst:
        dst.write(src.read())
    os.remove(bz2_path)


def _load():
    global _detector, _predictor
    if _predictor is None:
        _ensure_model()
        _detector = dlib.get_frontal_face_detector()
        _predictor = dlib.shape_predictor(MODEL_PATH)
    return _detector, _predictor


# --------------------------------------------------------------------------- #
# Hairstyle recommendations per shape
# --------------------------------------------------------------------------- #
SHAPE_TIPS = {
    "Oval": "The most versatile shape — almost anything works. Try long layers, "
            "a textured crop, or a side part. Avoid heavy fringes that hide the face.",
    "Round": "Add height and angles to lengthen the face — pompadour, quiff, or "
             "side-swept styles. Avoid rounded, full-volume cuts that add width.",
    "Square": "Soften the strong jaw — textured crops, side-swept fringe, and "
              "layered styles. Avoid very sharp, boxy cuts.",
    "Rectangle": "Add width, not height — fringes, side parts, and medium lengths "
                 "balance a long face. Avoid tall styles that elongate further.",
    "Oblong": "Reduce length and add sides — a fringe and medium, fuller sides "
              "shorten the face. Avoid long, height-adding styles.",
    "Heart": "Balance a wider forehead and narrow chin — side-swept fringe, "
             "chin-length, or layered medium styles. Avoid slicked-back looks.",
    "Diamond": "Add width at the forehead and jaw — fringes, side parts, and "
               "chin-length styles. Avoid styles that pull tight at the temples.",
    "Triangle (Pear)": "Add volume up top to balance a wider jaw — quiffs, layered "
                       "crowns, and fuller tops. Avoid flat, tight-on-top styles.",
    "Unknown": "Couldn't classify confidently — try a clearer, front-facing, "
               "well-lit photo.",
}


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _dist(p, q) -> float:
    return float(math.hypot(p[0] - q[0], p[1] - q[1]))


def _angle(a, b, c) -> float:
    """Angle (degrees) at vertex b formed by points a-b-c."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    denom = (math.hypot(*ba) * math.hypot(*bc)) + 1e-6
    cos = (ba[0] * bc[0] + ba[1] * bc[1]) / denom
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _measure(pts, rect):
    """Compute normalised facial measurements from the 68 landmarks."""
    chin = pts[8]
    forehead_w = _dist(pts[17], pts[26])          # outer brow to outer brow
    cheek_w = _dist(pts[1], pts[15])              # cheekbone width
    jaw_w = _dist(pts[4], pts[12])               # jaw (gonial) width
    # Estimate the hairline (forehead top) from the landmarks via the rule of thirds:
    # the forehead (hairline->brow) is ~half the brow->chin distance. dlib's HOG box
    # stops around the eyebrows, so we extrapolate above the highest brow point rather
    # than using the box top (which made the length line fall short of the forehead).
    brow_top_y = min(pts[i][1] for i in range(17, 27))
    top_y = int(max(brow_top_y - 0.5 * (chin[1] - brow_top_y), 0))
    length = _dist(chin, (chin[0], top_y))

    # jaw angularity: average gonial angle on both sides (smaller = more angular)
    jaw_angle = (_angle(pts[2], pts[4], pts[6]) + _angle(pts[14], pts[12], pts[10])) / 2.0

    return {
        "forehead_w": forehead_w,
        "cheek_w": cheek_w,
        "jaw_w": jaw_w,
        "length": length,
        "top_y": top_y,
        "jaw_angle": jaw_angle,
        "len_to_width": length / max(cheek_w, 1e-6),
        "forehead_to_cheek": forehead_w / max(cheek_w, 1e-6),
        "jaw_to_cheek": jaw_w / max(cheek_w, 1e-6),
    }


def classify(m) -> str:
    """
    Rule-based classifier using normalised ratios, calibrated against real-landmark
    baselines (an average face has jaw/cheek ~0.82 and forehead/cheek ~0.85, using
    the eyebrow span as the forehead proxy). Primary axes are the length/width ratio
    and jaw angularity (both reliable from dlib's 68 points); Heart/Diamond/Triangle
    are treated as special cases that need a clearly tapered or inverted jaw.
    """
    fore_ratio = m["forehead_to_cheek"]   # ~0.85 baseline for an average face
    jaw_ratio = m["jaw_to_cheek"]         # ~0.82 baseline for an average face
    ratio = m["len_to_width"]             # face length / cheekbone width
    angular = m["jaw_angle"] < 150.0      # smaller gonial angle => more angular jaw

    # Triangle / Pear: jaw is wider than the cheekbones (inverse taper)
    if jaw_ratio > 1.0:
        return "Triangle (Pear)"

    # Strong jaw taper (clearly narrow jaw / pointed lower face) -> Heart vs Diamond
    if jaw_ratio < 0.75:
        return "Heart" if fore_ratio > 0.82 else "Diamond"

    # Normal taper -> decide by face length (chin-to-hairline / cheekbone width) and
    # jaw angularity. Thresholds calibrated for the hairline-inclusive length:
    #   short (<1.30): Round / Square   mid (1.30-1.60): Oval / Rectangle   long: Oblong / Rectangle
    if ratio < 1.30:
        return "Square" if angular else "Round"
    if ratio < 1.60:
        return "Rectangle" if angular else "Oval"
    return "Rectangle" if angular else "Oblong"


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _annotate(image, pts, m, rect, shape):
    out = image.copy()
    chin = pts[8]
    # landmark dots
    for (x, y) in pts:
        cv2.circle(out, (int(x), int(y)), 2, (0, 200, 255), -1)
    # measurement lines
    green = (0, 220, 0)
    cv2.line(out, tuple(pts[17]), tuple(pts[26]), green, 2)      # forehead
    cv2.line(out, tuple(pts[1]), tuple(pts[15]), green, 2)       # cheekbone
    cv2.line(out, tuple(pts[4]), tuple(pts[12]), green, 2)       # jaw
    cv2.line(out, (chin[0], m["top_y"]), tuple(chin), green, 2)  # length (to estimated hairline)
    # label
    cv2.putText(out, shape, (rect.left(), max(rect.top() - 12, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def analyze(image_bgr):
    """
    Run the full pipeline on a BGR image (numpy array).
    Returns dict: {ok, shape, tip, measurements, annotated (BGR)} or {ok: False, error}.
    """
    detector, predictor = _load()

    # downscale only if very large (keeps native aspect ratio — no squishing)
    h, w = image_bgr.shape[:2]
    if max(h, w) > 1000:
        scale = 1000.0 / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        return {"ok": False, "error": "No face detected. Try a clearer, front-facing photo."}

    # use the largest detected face
    rect = max(faces, key=lambda r: r.width() * r.height())
    shape68 = predictor(gray, rect)
    pts = [(p.x, p.y) for p in shape68.parts()]

    m = _measure(pts, rect)
    shape = classify(m)
    annotated = _annotate(image_bgr, pts, m, rect, shape)

    return {
        "ok": True,
        "shape": shape,
        "tip": SHAPE_TIPS.get(shape, SHAPE_TIPS["Unknown"]),
        "measurements": m,
        "annotated": annotated,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python face_shape.py <image>")
        raise SystemExit(1)
    img = cv2.imread(sys.argv[1])
    res = analyze(img)
    if not res["ok"]:
        print(res["error"])
    else:
        print("Face shape:", res["shape"])
        for k, v in res["measurements"].items():
            print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")
        cv2.imwrite("annotated_output.jpg", res["annotated"])
        print("Saved annotated_output.jpg")
