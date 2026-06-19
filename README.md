---
title: Face Shape Detector
emoji: 🪞
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# 🪞 Face-Shape Detector

Detect a person's **face shape** from a single photo using classic computer vision —
**OpenCV** + **dlib** 81-point facial landmarks (with forehead/hairline points) — then recommend flattering hairstyles.

**▶️ Live demo:** _<add your Hugging Face Space URL here>_

![demo](assets/demo.jpg)

---

## What it does

Upload a front-facing photo and the app:

1. Detects the face (dlib HOG detector) and maps **81 facial landmarks** (including the forehead/hairline).
2. Measures forehead, cheekbone, and jaw widths, face length, and jaw angle.
3. Converts those to **normalised ratios** and classifies into one of **8 shapes**.
4. Draws the landmarks + measurement lines and returns a **hairstyle recommendation**.

## How it works

```
photo ──▶ dlib face detector ──▶ 81 landmarks ──▶ geometric measurements
                                                        │
                              normalised ratios + jaw angle
                                                        │
                                                  rule-based classifier ──▶ face shape + hairstyle tips
```

**Measurements** (all from the 81 landmarks, kept comparable and normalised):

| Feature | From landmarks |
|--------|----------------|
| Forehead width | hairline landmarks (68–80) |
| Cheekbone width | 1 ↔ 15 |
| Jaw width | 4 ↔ 12 |
| Face length | chin (8) → hairline (highest forehead landmark) |
| Jaw angle | gonial angle at 2-4-6 / 10-12-14 |

**Shapes covered:** Oval · Round · Square · Rectangle · Oblong · Heart · Diamond · Triangle (Pear).

## Engineering notes (what I improved)

This started as a script that classified **almost everyone as "oblong."** I rebuilt the
classification:

- **Diagnosed the bug:** the original compared **absolute pixel thresholds** on an image
  that was **resized to a fixed 500×500** — which squished the aspect ratio and corrupted
  the very length-vs-width proportions the logic depended on. "Oblong" had become the
  fall-through bucket.
- **Switched to normalised ratios** (length/width, region-to-region width ratios, jaw
  angle) so it generalises across image sizes and framing.
- **Stopped squishing the aspect ratio** — only downscale very large images.
- **Solved the "no forehead landmarks" problem** by upgrading to dlib's **81-point model**,
  which adds 13 real forehead/hairline points (68–80) — so forehead width *and* face length
  are measured from the **actual hairline**, not a fragile KMeans hair/skin segmentation or
  an eyebrow/temple proxy.
- **Calibrated the thresholds against real landmark geometry** so a typical face lands on a
  common shape (Oval/Round) instead of collapsing to one class — and **added the missing
  Oval and Heart shapes**.

> Note: face-shape classification from 2D landmarks is a **heuristic**, not exact science.
> The goal here is a transparent, well-calibrated demo — not clinical accuracy.

## Run locally

```bash
pip install -r requirements.txt
python app.py            # launches the Gradio web UI
# or, headless on one image:
python face_shape.py path/to/photo.jpg   # prints the shape + saves annotated_output.jpg
```

The dlib 81-landmark model (~18 MB) is downloaded automatically on first run.

## Tech stack

Python · OpenCV · dlib · NumPy · Gradio · deployed on Hugging Face Spaces.

## Credits

- Landmark model: [`shape_predictor_81_face_landmarks`](https://github.com/codeniko/shape_predictor_81_face_landmarks) (extends dlib's 68-point model with forehead landmarks; non-commercial / iBUG license).
- Example faces are AI-generated (StyleGAN2 / thispersondoesnotexist) — not real people.

## License

MIT
