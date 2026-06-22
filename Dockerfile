# Container image for the Gradio face-shape detector.
# Used by Render (and any Docker host). HF Spaces ignores this file — it uses the
# README `sdk: gradio` runtime instead.
FROM python:3.12-slim

# System libs: OpenCV-headless needs libglib2.0-0; dlib needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + bundled 81-point model (pulled via Git LFS by the host on clone).
COPY . .

# Render injects $PORT at runtime; default 7860 for local/other hosts.
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
