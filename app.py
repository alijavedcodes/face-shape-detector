"""
Gradio web demo for the face-shape detector.

Upload a front-facing photo -> the app runs the OpenCV + dlib pipeline and shows
the detected landmarks, the classified face shape, and a hairstyle recommendation.

Run locally:   python app.py
Deploy:        push to a Hugging Face Space (SDK: gradio) — see README.
"""
import cv2
import gradio as gr

from face_shape import analyze


def run(image_rgb):
    if image_rgb is None:
        return None, {"No image": 1.0}, "Upload a front-facing photo to begin."

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    res = analyze(bgr)

    if not res["ok"]:
        return None, {"No face detected": 1.0}, f"⚠️ {res['error']}"

    annotated_rgb = cv2.cvtColor(res["annotated"], cv2.COLOR_BGR2RGB)
    m = res["measurements"]
    scores = res["scores"]
    blend = ", ".join(f"{k} {v * 100:.0f}%" for k, v in list(scores.items())[:2])
    details = (
        f"### {res['shape']} · {scores[res['shape']] * 100:.0f}%\n"
        f"{res['tip']}\n\n"
        f"**Closest shapes:** {blend}  \n"
        f"*Face shape is subjective (people disagree ~25–30%), so this shows a blend, not a verdict.*\n\n"
        f"**Measurements (normalised ratios)**\n"
        f"- Length / cheekbone width: `{m['len_to_width']:.2f}`\n"
        f"- Forehead / cheekbone: `{m['forehead_to_cheek']:.2f}`\n"
        f"- Jaw / cheekbone: `{m['jaw_to_cheek']:.2f}`\n"
        f"- Jaw angle: `{m['jaw_angle']:.0f}°`\n"
    )
    return annotated_rgb, scores, details


DESCRIPTION = """
# 🪞 Face-Shape Detector
Detects your face shape from a single photo using **OpenCV** (image handling) and
**dlib** (HOG face detector + 68-point facial landmarks), then recommends hairstyles.

It measures forehead, cheekbone, and jaw widths plus face length and jaw angle,
converts them to **normalised ratios**, and classifies into one of 8 shapes
(Oval, Round, Square, Rectangle, Oblong, Heart, Diamond, Triangle).

*Tip: use a clear, front-facing, well-lit photo. Faces only — no real personal data is stored.*
"""

with gr.Blocks(title="Face-Shape Detector") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Upload a front-facing photo")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Detected landmarks & measurements")
            out_shape = gr.Label(label="Face shape (confidence blend)", num_top_classes=3)
            out_text = gr.Markdown()
    gr.Examples(
        examples=[
            ["sample_images/sample1.jpg"],
            ["sample_images/sample2.jpg"],
            ["sample_images/sample3.jpg"],
            ["sample_images/sample4.jpg"],
            ["sample_images/sample5.jpg"],
        ],
        inputs=inp,
        outputs=[out_img, out_shape, out_text],
        fn=run,
        cache_examples=False,
        label="Example faces (AI-generated — thispersondoesnotexist / StyleGAN2)",
    )
    btn.click(run, inputs=inp, outputs=[out_img, out_shape, out_text])
    inp.upload(run, inputs=inp, outputs=[out_img, out_shape, out_text])

if __name__ == "__main__":
    demo.launch()
