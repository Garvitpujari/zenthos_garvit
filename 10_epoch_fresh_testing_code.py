# install commands
!pip install -U packaging ninja einops
!pip install -U mamba-ssm --no-build-isolation
pip install gradio
!pip install opencv-python
!pip uninstall -y accelerate transformers bitsandbytes
!pip install --no-cache-dir accelerate==0.26.1 transformers==4.38.2

# ============================================================
# 🎥 GRADIO INFERENCE
# CLIP ViT-L/14 (image + text) + MAMBA (temporal)
# ============================================================

import torch
import torch.nn as nn
import gradio as gr
import cv2
import numpy as np
import boto3
import io

from transformers import CLIPModel, CLIPProcessor
from mamba_ssm import Mamba

# ===================== CONFIG =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

S3_BUCKET = "tensorflow-file-of-charades-dataset"
CHECKPOINT_KEY = "mamba_charades_10epochs/epoch_5.pth"

FRAME_STRIDE = 8   # sample every N frames

# ===================== LOAD CLIP =====================
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

clip.eval()
for p in clip.parameters():
    p.requires_grad = False

# ===================== MAMBA VIDEO ENCODER =====================
class MambaVideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = Mamba(
            d_model=768,
            d_state=16,
            d_conv=4,
            expand=2
        )

    def forward(self, frame_embs):
        # frame_embs: (T, 768)
        x = frame_embs.unsqueeze(0)   # (1, T, 768)
        y = self.mamba(x)
        y = y[:, -1]                  # full video summary
        return nn.functional.normalize(y, dim=-1)

model = MambaVideoEncoder().to(DEVICE)
model.eval()

# ===================== LOAD TRAINED WEIGHTS =====================
print("⬇️ Loading trained Mamba checkpoint from S3...")
s3 = boto3.client("s3")
buf = io.BytesIO()
s3.download_fileobj(S3_BUCKET, CHECKPOINT_KEY, buf)
buf.seek(0)

ckpt = torch.load(buf, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
print("✅ Model loaded (epoch 5)")

# ===================== VIDEO → FRAME EMBEDDINGS =====================
def video_to_frame_embeddings(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % FRAME_STRIDE == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        idx += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames extracted from video")

    inputs = processor(
        images=frames,
        return_tensors="pt"
    )

    with torch.no_grad():
        img_embs = clip.get_image_features(**inputs.to(DEVICE))
        img_embs = nn.functional.normalize(img_embs, dim=-1)

    return img_embs   # (T, 768)

# ===================== INFERENCE FUNCTION =====================
def run_inference(video, text_block):
    if video is None or text_block.strip() == "":
        return "❌ Please upload a video and enter at least one sentence."

    # Parse sentences (one per line)
    sentences = [s.strip() for s in text_block.split("\n") if s.strip()]
    if len(sentences) == 0:
        return "❌ No valid sentences found."

    # Video → frame embeddings → Mamba
    frame_embs = video_to_frame_embeddings(video)
    with torch.no_grad():
        video_emb = model(frame_embs)   # (1, 768)

    # Encode text (NO hardcoding, any number)
    text_inputs = processor(
        text=sentences,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        text_embs = clip.get_text_features(**text_inputs.to(DEVICE))
        text_embs = nn.functional.normalize(text_embs, dim=-1)

    # Similarity
    sims = (video_emb @ text_embs.T).squeeze(0)

    # Prepare output (no scrolling, all visible)
    output_lines = []
    for sent, score in zip(sentences, sims.tolist()):
        output_lines.append(f"{sent}  ➜  similarity: {score:.4f}")

    return "\n".join(output_lines)

# ===================== GRADIO UI =====================
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Video–Text Similarity (CLIP + Mamba)")
    gr.Markdown(
        "Upload a video and enter **any number of sentences** (one per line).  \n"
        "The model computes similarity scores in CLIP shared space."
    )

    with gr.Row():
        video_input = gr.Video(label="Upload Video")
        text_input = gr.Textbox(
            label="Enter sentences (one per line)",
            lines=10,
            placeholder="e.g.\nperson opens refrigerator\nperson sits on chair\nperson closes door"
        )

    output = gr.Textbox(
        label="Similarity Scores",
        lines=15
    )

    btn = gr.Button("Run Inference")
    btn.click(run_inference, inputs=[video_input, text_input], outputs=output)
demo.launch(share=True)

