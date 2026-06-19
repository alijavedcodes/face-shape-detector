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
MODEL_PATH = os.path.join(MODEL_DIR, "shape_predictor_81_face_landmarks.dat")
# 81-point model = the standard 68 landmarks PLUS 13 forehead/hairline points (68-80).
MODEL_URL = (
    "https://github.com/codeniko/shape_predictor_81_face_landmarks/raw/master/"
    "shape_predictor_81_face_landmarks.dat"
)

_detector = None
_predictor = None


def _ensure_model() -> None:
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading dlib 81-point landmark model (~18 MB, one time)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


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
    """Compute normalised facial measurements from the 81 landmarks."""
    chin = pts[8]

    # Forehead (81-point hairline landmarks 68-80): the outer points sit low at the
    # temples, so (a) trim the 3 outermost points each side for the WIDTH so it stays
    # within the forehead skin (not into the hair), and (b) draw the line at the vertical
    # CENTRE of the forehead (midway between the brow line and the hairline).
    forehead_pts = pts[68:81]
    xs = sorted(p[0] for p in forehead_pts)
    ys = sorted(p[1] for p in forehead_pts)
    fx_left, fx_right = xs[3], xs[-4]
    forehead_w = fx_right - fx_left
    top_y = ys[0]                                   # highest hairline point (face length)
    brow_top_y = min(pts[i][1] for i in range(17, 27))
    fore_line_y = int((brow_top_y + top_y) / 2)     # vertical centre of the forehead

    cheek_w = _dist(pts[1], pts[15])      # cheekbone width (widest face outline)
    jaw_w = _dist(pts[3], pts[13])        # jaw width = bigonial width at the jaw corner (gonial angle)

    # Face length: chin -> actual hairline (highest forehead landmark)
    length = _dist(chin, (chin[0], top_y))

    # Jaw angularity: average gonial angle on both sides (smaller = more angular)
    jaw_angle = (_angle(pts[2], pts[4], pts[6]) + _angle(pts[14], pts[12], pts[10])) / 2.0

    return {
        "forehead_w": forehead_w,
        "cheek_w": cheek_w,
        "jaw_w": jaw_w,
        "length": length,
        "top_y": top_y,
        "forehead_line": ((fx_left, fore_line_y), (fx_right, fore_line_y)),
        "jaw_angle": jaw_angle,
        "len_to_width": length / max(cheek_w, 1e-6),
        "forehead_to_cheek": forehead_w / max(cheek_w, 1e-6),
        "jaw_to_cheek": jaw_w / max(cheek_w, 1e-6),
    }


def _membership(m):
    """
    Soft per-shape scoring -> normalised confidences (dict, highest first).

    Face shape is a subjective label (humans disagree ~25-30%), so instead of a hard
    single class we score how well the measurements fit each shape and return a
    confidence blend. Built from smooth membership functions over the same calibrated
    features: length/width (L), jaw/cheek taper (J, the bigonial width at the jaw corner
    3<->13 so the normal band is ~0.87-0.95), forehead/cheek (F), and jaw angle (A).
    """
    L, J, F, A = m["len_to_width"], m["jaw_to_cheek"], m["forehead_to_cheek"], m["jaw_angle"]

    def bump(x, c, w):
        return math.exp(-((x - c) / w) ** 2)

    def hi(x, t, s):
        return 1.0 / (1.0 + math.exp(-(x - t) / s))

    def lo(x, t, s):
        return 1.0 / (1.0 + math.exp((x - t) / s))

    short, mid, long_ = lo(L, 1.30, 0.06), bump(L, 1.42, 0.14), hi(L, 1.55, 0.06)
    soft, angular = hi(A, 150, 5), lo(A, 150, 5)
    normaljaw = bump(J, 0.92, 0.06)
    narrowjaw, widejaw = lo(J, 0.86, 0.03), hi(J, 0.98, 0.03)
    wide_fore, narrow_fore = hi(F, 0.96, 0.04), lo(F, 0.92, 0.04)

    raw = {
        "Round": short * soft * normaljaw,
        "Square": short * angular * normaljaw,
        "Oval": mid * soft * normaljaw,
        "Oblong": long_ * soft * normaljaw,
        "Rectangle": long_ * angular * normaljaw,
        "Heart": narrowjaw * wide_fore,
        "Diamond": narrowjaw * narrow_fore,
        "Triangle (Pear)": widejaw,
    }
    total = sum(raw.values()) or 1.0
    conf = {k: v / total for k, v in raw.items()}
    return dict(sorted(conf.items(), key=lambda kv: kv[1], reverse=True))


def classify(m) -> str:
    """Top face-shape label (see _membership for the full confidence blend)."""
    return next(iter(_membership(m)))


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
    cv2.line(out, m["forehead_line"][0], m["forehead_line"][1], green, 2)  # forehead (temple width, raised onto forehead)
    cv2.line(out, tuple(pts[1]), tuple(pts[15]), green, 2)       # cheekbone
    cv2.line(out, tuple(pts[3]), tuple(pts[13]), green, 2)       # jaw (bigonial width at the jaw corner)
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
    scores = _membership(m)
    shape = next(iter(scores))
    annotated = _annotate(image_bgr, pts, m, rect, shape)

    return {
        "ok": True,
        "shape": shape,
        "scores": scores,
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
