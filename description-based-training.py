# ============================================================
# 🔁 DESCRIPTION-BASED FINETUNING (FROM epoch_40)
# EMBEDDINGS READY | GPU SAFE | TEXT ENCODING CHUNKED
# ============================================================

import os
import io
import torch
import torch.nn as nn
import pandas as pd
import boto3
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from mamba_ssm import Mamba

# ============================================================
# 0️⃣ CONFIG
# ============================================================
S3_BUCKET = "tensorflow-file-of-charades-dataset"
CHECKPOINT_PREFIX = "mamba_charades_retraining"

LOCAL_META_DIR = "./charades_data"
LOCAL_EMBED_DIR = "/tmp/charades_embeddings"

TRAIN_CSV = f"{LOCAL_META_DIR}/Charades_v1_train.csv"
VAL_CSV   = f"{LOCAL_META_DIR}/Charades_v1_test.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM = 768
LR = 1e-4
DROPOUT = 0.2
BATCH_SIZE = 4
FINE_TUNE_EPOCHS = 10
TEMPERATURE = 0.1
TEXT_CHUNK = 512          # for similarity
TEXT_ENCODE_CHUNK = 256   # for CLIP text encoder

BASE_EPOCH = 40

s3 = boto3.client("s3")

# ============================================================
# 1️⃣ LOAD CSVs
# ============================================================
train_df = pd.read_csv(TRAIN_CSV).rename(columns={"descriptions": "description"})
val_df   = pd.read_csv(VAL_CSV).rename(columns={"descriptions": "description"})

embed_vids = set(f.replace(".pt", "") for f in os.listdir(LOCAL_EMBED_DIR))
train_df = train_df[train_df["id"].isin(embed_vids)]
val_df   = val_df[val_df["id"].isin(embed_vids)]

train_vids = train_df["id"].tolist()
train_desc = train_df["description"].tolist()
val_vids   = val_df["id"].tolist()
val_desc   = val_df["description"].tolist()

ALL_DESCRIPTIONS = train_desc + val_desc

print(f"✅ Train videos: {len(train_vids)} | Val videos: {len(val_vids)}")
print(f"📝 Total descriptions: {len(ALL_DESCRIPTIONS)}")

# ============================================================
# 2️⃣ CLIP TEXT ENCODER (CPU ONLY, CHUNKED)
# ============================================================
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

clip.eval()
for p in clip.parameters():
    p.requires_grad = False

all_text_embeds = []

with torch.no_grad():
    for i in tqdm(range(0, len(ALL_DESCRIPTIONS), TEXT_ENCODE_CHUNK),
                  desc="Encoding text descriptions"):
        chunk = ALL_DESCRIPTIONS[i:i + TEXT_ENCODE_CHUNK]
        inputs = processor(
            text=chunk,
            return_tensors="pt",
  
         padding=True,
            truncation=True
        )
     emb = clip.get_text_features(**inputs)
        emb = nn.functional.normalize(emb, dim=-1)
        all_text_embeds.append(emb.cpu())

text_embeds = torch.cat(all_text_embeds, dim=0)
NUM_TEXTS = text_embeds.size(0)

print(f"✅ Text embeddings ready: {NUM_TEXTS}")

# ============================================================
# 3️⃣ MAMBA VIDEO ENCODER (ARCHITECTURE PRESERVED)
# ============================================================
class MambaVideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = Mamba(
            d_model=EMBED_DIM,
            d_state=16,       
            d_conv=4,   
            expand=2 
        )
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, frame_embs):
        """
        frame_embs: (T, 768)
        """
        x = frame_embs.unsqueeze(0)   # (1, T, 768)
        y = self.mamba(x)             # sequential processing
        y = self.dropout(y[:, -1])    # final hidden state
        return nn.functional.normalize(y, dim=-1)

model = MambaVideoEncoder().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# ============================================================
# 4️⃣ LOAD CHECKPOINT (epoch_40)
# ============================================================
ckpt_key = f"{CHECKPOINT_PREFIX}/epoch_{BASE_EPOCH}.pth"
ckpt_local = f"/tmp/epoch_{BASE_EPOCH}.pth"
s3.download_file(S3_BUCKET, ckpt_key, ckpt_local)

ckpt = torch.load(ckpt_local, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])

print(f"✅ Loaded checkpoint epoch_{BASE_EPOCH}")

# ============================================================
# 5️⃣ CHUNKED CONTRASTIVE LOSS (GPU SAFE)
# ============================================================
def contrastive_loss_chunked(video_embs, target_indices):
    B = video_embs.size(0)
    loss = 0.0

    for start in range(0, NUM_TEXTS, TEXT_CHUNK):
        end = min(start + TEXT_CHUNK, NUM_TEXTS)
        chunk = text_embeds[start:end].to(DEVICE)

        logits = (video_embs @ chunk.T) / TEMPERATURE
        log_probs = torch.log_softmax(logits, dim=1)

        for i in range(B):
            tgt = target_indices[i]
            if start <= tgt < end:
                loss -= log_probs[i, tgt - start]

        del chunk, logits, log_probs

    return loss / B

def load_video_embeddings(vid):
    return torch.load(f"{LOCAL_EMBED_DIR}/{vid}.pt").to(DEVICE).float()

# ============================================================
# 6️⃣ TRAIN / VAL EPOCH
# ============================================================
def run_epoch(video_ids, text_offset, train=True):
    model.train() if train else model.eval()
    total_loss, count = 0.0, 0

    for i in tqdm(range(0, len(video_ids), BATCH_SIZE), leave=False):
        batch_vids = video_ids[i:i+BATCH_SIZE]

        video_embs = []
        target_indices = []

        for j, vid in enumerate(batch_vids):
            frames = load_video_embeddings(vid)   # (T, 768)
            emb = model(frames)                   # single forward
            video_embs.append(emb)
            target_indices.append(text_offset + i + j)

        video_embs = torch.cat(video_embs, dim=0)
        loss = contrastive_loss_chunked(video_embs, target_indices)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * video_embs.size(0)
        count += video_embs.size(0)

    return total_loss / count

# ============================================================
# 7️⃣ FINETUNING LOOP
# ============================================================
for epoch in range(1, FINE_TUNE_EPOCHS + 1):
    train_loss = run_epoch(train_vids, text_offset=0, train=True)
    val_loss   = run_epoch(val_vids, text_offset=len(train_vids), train=False)

    print(f"[Desc-Finetune] Epoch {epoch} | train={train_loss:.4f} | val={val_loss:.4f}")

    buf = io.BytesIO()
    torch.save({"model_state": model.state_dict()}, buf)
    buf.seek(0)

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"{CHECKPOINT_PREFIX}/epoch_{BASE_EPOCH + epoch}.pth",
        Body=buf.getvalue()
    )

print("🎉 DESCRIPTION-BASED FINETUNING COMPLETE")
