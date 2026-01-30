# ============================================================
# REQUIRED CONFIG (MUST BE DEFINED BEFORE EMBEDDING SYNC)
# ============================================================

import os
import boto3
from tqdm import tqdm

S3_BUCKET = "tensorflow-file-of-charades-dataset"
EMBED_PREFIX = "charades_clip_embeddings_final"

LOCAL_EMBED_DIR = "/tmp/charades_embeddings"
os.makedirs(LOCAL_EMBED_DIR, exist_ok=True)

s3 = boto3.client("s3")



# ============================================================
# 2️⃣ SYNC ALL FRAME EMBEDDINGS FROM S3 (WITH PROGRESS)
# ============================================================
print("⬇️ Syncing frame-level CLIP embeddings from S3...")

# First pass: count total embeddings in S3
total_s3_embeddings = 0
continuation = None

while True:
    kwargs = dict(Bucket=S3_BUCKET, Prefix=EMBED_PREFIX)
    if continuation:
        kwargs["ContinuationToken"] = continuation

    resp = s3.list_objects_v2(**kwargs)
    for obj in resp.get("Contents", []):
        if obj["Key"].endswith(".pt"):
            total_s3_embeddings += 1

    if resp.get("IsTruncated"):
        continuation = resp["NextContinuationToken"]
    else:
        break

print(f"📦 Total embeddings available on S3: {total_s3_embeddings}")

# Second pass: download missing embeddings with progress bar
downloaded_embeddings = 0
existing_embeddings = set(os.listdir(LOCAL_EMBED_DIR))

with tqdm(total=total_s3_embeddings, desc="Downloading embeddings") as pbar:
    continuation = None
    while True:
        kwargs = dict(Bucket=S3_BUCKET, Prefix=EMBED_PREFIX)
        if continuation:
            kwargs["ContinuationToken"] = continuation

        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".pt"):
                continue

            fname = key.split("/")[-1]
            local_path = f"{LOCAL_EMBED_DIR}/{fname}"

            if fname not in existing_embeddings:
                s3.download_file(S3_BUCKET, key, local_path)
                downloaded_embeddings += 1

            pbar.update(1)

        if resp.get("IsTruncated"):
            continuation = resp["NextContinuationToken"]
        else:
            break

# Final sanity set
EMBED_VIDS = set(f.replace(".pt", "") for f in os.listdir(LOCAL_EMBED_DIR))

print(
    f"✅ Embeddings ready: {len(EMBED_VIDS)} / {total_s3_embeddings} "
    f"(downloaded this run: {downloaded_embeddings})"
)
# this above was the code to just load the embeddings from s3 with progress bar

# code for finetuning for 20 mmore epochs starts from here loadigs the weights from epoch 10 



# # ============================================================
# # 🎯 VIDEO-LEVEL DESCRIPTION FINETUNING (EPOCH 21 → 40)
# # CLIP ViT-L/14 (FROZEN) + MAMBA TEMPORAL SUMMARY
# # ============================================================

# import os
# import io
# import torch
# import torch.nn as nn
# import pandas as pd
# import boto3
# from tqdm import tqdm
# from transformers import CLIPModel, CLIPProcessor
# from mamba_ssm import Mamba

# # ============================================================
# # 0️⃣ CONFIG
# # ============================================================
# S3_BUCKET = "tensorflow-file-of-charades-dataset"
# CHECKPOINT_PREFIX = "mamba_charades_10epochs"

# S3_META_PREFIX = "charades_data"
# EMBED_PREFIX = "charades_clip_embeddings_final"

# LOCAL_META_DIR = "./charades_data"
# LOCAL_EMBED_DIR = "/tmp/charades_embeddings"
# os.makedirs(LOCAL_META_DIR, exist_ok=True)
# os.makedirs(LOCAL_EMBED_DIR, exist_ok=True)

# TRAIN_CSV = f"{LOCAL_META_DIR}/Charades_v1_train.csv"
# VAL_CSV   = f"{LOCAL_META_DIR}/Charades_v1_test.csv"

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# EMBED_DIM = 768
# LR = 1e-5
# DROPOUT = 0.2
# BATCH_SIZE = 4
# TEMPERATURE = 0.1

# TEXT_ENCODE_CHUNK = 256
# TEXT_SIM_CHUNK = 512

# RESUME_EPOCH = 20
# FINETUNE_EPOCHS = 20   # 21 → 40

# s3 = boto3.client("s3")

# # ============================================================
# # 1️⃣ DOWNLOAD METADATA (IF MISSING)
# # ============================================================
# def download_if_missing(key, local_path):
#     if not os.path.exists(local_path):
#         s3.download_file(S3_BUCKET, key, local_path)

# download_if_missing(f"{S3_META_PREFIX}/Charades_v1_train.csv", TRAIN_CSV)
# download_if_missing(f"{S3_META_PREFIX}/Charades_v1_test.csv", VAL_CSV)

# # ============================================================
# # 2️⃣ SYNC ALL FRAME EMBEDDINGS FROM S3
# # ============================================================
# print("⬇️ Syncing frame-level CLIP embeddings from S3...")

# total_s3_embeddings = 0
# continuation = None
# while True:
#     kwargs = dict(Bucket=S3_BUCKET, Prefix=EMBED_PREFIX)
#     if continuation:
#         kwargs["ContinuationToken"] = continuation
#     resp = s3.list_objects_v2(**kwargs)
#     for obj in resp.get("Contents", []):
#         if obj["Key"].endswith(".pt"):
#             total_s3_embeddings += 1
#     if resp.get("IsTruncated"):
#         continuation = resp["NextContinuationToken"]
#     else:
#         break

# downloaded = 0
# existing = set(os.listdir(LOCAL_EMBED_DIR))

# with tqdm(total=total_s3_embeddings, desc="Downloading embeddings") as pbar:
#     continuation = None
#     while True:
#         kwargs = dict(Bucket=S3_BUCKET, Prefix=EMBED_PREFIX)
#         if continuation:
#             kwargs["ContinuationToken"] = continuation
#         resp = s3.list_objects_v2(**kwargs)
#         for obj in resp.get("Contents", []):
#             key = obj["Key"]
#             if not key.endswith(".pt"):
#                 continue
#             fname = key.split("/")[-1]
#             if fname not in existing:
#                 s3.download_file(S3_BUCKET, key, f"{LOCAL_EMBED_DIR}/{fname}")
#                 downloaded += 1
#             pbar.update(1)
#         if resp.get("IsTruncated"):
#             continuation = resp["NextContinuationToken"]
#         else:
#             break

# EMBED_VIDS = set(f.replace(".pt", "") for f in os.listdir(LOCAL_EMBED_DIR))

# print(f"✅ Embeddings ready: {len(EMBED_VIDS)} / {total_s3_embeddings} "
#       f"(downloaded this run: {downloaded})")

# # ============================================================
# # 3️⃣ LOAD CSVs (FILTERED)
# # ============================================================
# train_df = pd.read_csv(TRAIN_CSV).rename(columns={"descriptions": "description"})
# val_df   = pd.read_csv(VAL_CSV).rename(columns={"descriptions": "description"})

# train_df = train_df[train_df["id"].isin(EMBED_VIDS)]
# val_df   = val_df[val_df["id"].isin(EMBED_VIDS)]

# train_vids = train_df["id"].tolist()
# train_desc = train_df["description"].tolist()
# val_vids   = val_df["id"].tolist()
# val_desc   = val_df["description"].tolist()

# ALL_DESCRIPTIONS = train_desc + val_desc

# print(f"✅ Train videos: {len(train_vids)} | Val videos: {len(val_vids)}")
# print(f"📝 Total descriptions: {len(ALL_DESCRIPTIONS)}")

# # ============================================================
# # 4️⃣ CLIP TEXT ENCODER (CPU, FROZEN)
# # ============================================================
# clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# clip.eval()
# for p in clip.parameters():
#     p.requires_grad = False

# all_text_embeds = []
# with torch.no_grad():
#     for i in tqdm(range(0, len(ALL_DESCRIPTIONS), TEXT_ENCODE_CHUNK),
#                   desc="Encoding text descriptions"):
#         chunk = ALL_DESCRIPTIONS[i:i + TEXT_ENCODE_CHUNK]
#         inputs = processor(text=chunk, return_tensors="pt",
#                            padding=True, truncation=True)
#         emb = clip.get_text_features(**inputs)
#         emb = nn.functional.normalize(emb, dim=-1)
#         all_text_embeds.append(emb.cpu())

# text_embeds = torch.cat(all_text_embeds, dim=0)
# NUM_TEXTS = text_embeds.size(0)

# print(f"✅ Text embeddings ready: {NUM_TEXTS}")

# # ============================================================
# # 5️⃣ MAMBA VIDEO ENCODER (ARCHITECTURE)
# # ============================================================
# class MambaVideoEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mamba = Mamba(
#             d_model=EMBED_DIM,
#             d_state=16,
#             d_conv=4,
#             expand=2
#         )
#         self.dropout = nn.Dropout(DROPOUT)

#     def forward(self, frame_embs):
#         x = frame_embs.unsqueeze(0)      # (1, T, 768)
#         y = self.mamba(x)
#         y = self.dropout(y[:, -1])       # last state = video summary
#         return nn.functional.normalize(y, dim=-1)

# model = MambaVideoEncoder().to(DEVICE)
# optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

# # ============================================================
# # 6️⃣ LOAD CHECKPOINT (EPOCH 10)
# # ============================================================
# print("⬇️ Loading epoch_20 checkpoint from S3...")

# ckpt_buf = io.BytesIO()
# s3.download_fileobj(
#     Bucket=S3_BUCKET,
#     Key=f"{CHECKPOINT_PREFIX}/epoch_{RESUME_EPOCH}.pth",
#     Fileobj=ckpt_buf
# )
# ckpt_buf.seek(0)

# checkpoint = torch.load(ckpt_buf, map_location=DEVICE)
# model.load_state_dict(checkpoint["model_state"])
# optimizer.load_state_dict(checkpoint["optimizer_state"])

# print(f"✅ Resumed from epoch {checkpoint['epoch']}")

# # ============================================================
# # 7️⃣ UTILITIES
# # ============================================================
# def load_video_embeddings(vid):
#     return torch.load(f"{LOCAL_EMBED_DIR}/{vid}.pt").to(DEVICE).float()

# def contrastive_loss_chunked(video_embs, target_indices):
#     B = video_embs.size(0)
#     loss = 0.0
#     for start in range(0, NUM_TEXTS, TEXT_SIM_CHUNK):
#         end = min(start + TEXT_SIM_CHUNK, NUM_TEXTS)
#         chunk = text_embeds[start:end].to(DEVICE)
#         logits = (video_embs @ chunk.T) / TEMPERATURE
#         log_probs = torch.log_softmax(logits, dim=1)
#         for i in range(B):
#             tgt = target_indices[i]
#             if start <= tgt < end:
#                 loss -= log_probs[i, tgt - start]
#         del chunk, logits, log_probs
#     return loss / B

# # ============================================================
# # 8️⃣ TRAIN / VAL LOOP
# # ============================================================
# def run_epoch(video_ids, text_offset, train=True):
#     model.train() if train else model.eval()
#     total_loss, count = 0.0, 0

#     for i in tqdm(range(0, len(video_ids), BATCH_SIZE), leave=False):
#         batch_vids = video_ids[i:i + BATCH_SIZE]
#         video_embs, target_indices = [], []

#         for j, vid in enumerate(batch_vids):
#             frames = load_video_embeddings(vid)
#             emb = model(frames)
#             video_embs.append(emb)
#             target_indices.append(text_offset + i + j)

#         video_embs = torch.cat(video_embs, dim=0)
#         loss = contrastive_loss_chunked(video_embs, target_indices)

#         if train:
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()

#         total_loss += loss.item() * video_embs.size(0)
#         count += video_embs.size(0)

#     return total_loss / count

# # ============================================================
# # 9️⃣ FINETUNING (EPOCHS 21 → 40)
# # ============================================================
# for epoch in range(RESUME_EPOCH + 1, RESUME_EPOCH + FINETUNE_EPOCHS + 1):
#     train_loss = run_epoch(train_vids, text_offset=0, train=True)
#     val_loss = run_epoch(val_vids, text_offset=len(train_vids), train=False)

#     print(f"[Epoch {epoch}] train={train_loss:.4f} | val={val_loss:.4f}")

#     buf = io.BytesIO()
#     torch.save(
#         {
#             "epoch": epoch,
#             "model_state": model.state_dict(),
#             "optimizer_state": optimizer.state_dict()
#         },
#         buf
#     )
#     buf.seek(0)

#     s3.put_object(
#         Bucket=S3_BUCKET,
#         Key=f"{CHECKPOINT_PREFIX}/epoch_{epoch}.pth",
#         Body=buf.getvalue()
#     )

# print("🎉 FINETUNING COMPLETE — EPOCHS 21–40 SAVED")




# ============================================================
# 🎯 VIDEO-LEVEL DESCRIPTION FINETUNING (EPOCH 21 → 40)
# CLIP ViT-L/14 (FROZEN) + MAMBA TEMPORAL SUMMARY
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
CHECKPOINT_PREFIX = "mamba_charades_10epochs"

S3_META_PREFIX = "charades_data"
EMBED_PREFIX = "charades_clip_embeddings_final"

LOCAL_META_DIR = "./charades_data"
LOCAL_EMBED_DIR = "/tmp/charades_embeddings"
os.makedirs(LOCAL_META_DIR, exist_ok=True)
os.makedirs(LOCAL_EMBED_DIR, exist_ok=True)

TRAIN_CSV = f"{LOCAL_META_DIR}/Charades_v1_train.csv"
VAL_CSV   = f"{LOCAL_META_DIR}/Charades_v1_test.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM = 768
LR = 1e-5
DROPOUT = 0.2
BATCH_SIZE = 4
TEMPERATURE = 0.1

TEXT_ENCODE_CHUNK = 256
TEXT_SIM_CHUNK = 512

RESUME_EPOCH = 20
FINETUNE_EPOCHS = 20   # 21 → 40

s3 = boto3.client("s3")

# ============================================================
# 1️⃣ DOWNLOAD METADATA (IF MISSING)
# ============================================================
def download_if_missing(key, local_path):
    if not os.path.exists(local_path):
        s3.download_file(S3_BUCKET, key, local_path)

download_if_missing(f"{S3_META_PREFIX}/Charades_v1_train.csv", TRAIN_CSV)
download_if_missing(f"{S3_META_PREFIX}/Charades_v1_test.csv", VAL_CSV)

# ============================================================
# 2️⃣ SYNC ALL FRAME EMBEDDINGS FROM S3
# ============================================================
print("⬇️ Syncing frame-level CLIP embeddings from S3...")

total_s3_embeddings = 0
continuation = None
while True:
    kwargs = dict(Bucket=S3_BUCKET, Prefix=EMBED_PREFIX)
    if continuation:
        kwargs["ContinuationToken"] = continuation
    resp = s3.list_objects_v2(**kwargs)
    for obj in resp.get("Contents", []):
        if obj["Key"].endswith(".pt"):
            total_s3_embeddings += 1
    if resp.get("IsTruncated"):
        continuation = resp["NextContinuationToken"]
    else:
        break

downloaded = 0
existing = set(os.listdir(LOCAL_EMBED_DIR))

with tqdm(total=total_s3_embeddings, desc="Downloading embeddings") as pbar:
    continuation = None
    while True:
        kwargs = dict(Bucket=S3_BUCKET, Prefix=EMBED_PREFIX)
        if continuation:
            kwargs["ContinuationToken"] = continuation
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".pt"):
                continue
            fname = key.split("/")[-1]
            if fname not in existing:
                s3.download_file(S3_BUCKET, key, f"{LOCAL_EMBED_DIR}/{fname}")
                downloaded += 1
            pbar.update(1)
        if resp.get("IsTruncated"):
            continuation = resp["NextContinuationToken"]
        else:
            break

EMBED_VIDS = set(f.replace(".pt", "") for f in os.listdir(LOCAL_EMBED_DIR))

print(f"✅ Embeddings ready: {len(EMBED_VIDS)} / {total_s3_embeddings} "
      f"(downloaded this run: {downloaded})")

# ============================================================
# 3️⃣ LOAD CSVs (FILTERED)
# ============================================================
train_df = pd.read_csv(TRAIN_CSV).rename(columns={"descriptions": "description"})
val_df   = pd.read_csv(VAL_CSV).rename(columns={"descriptions": "description"})

train_df = train_df[train_df["id"].isin(EMBED_VIDS)]
val_df   = val_df[val_df["id"].isin(EMBED_VIDS)]

train_vids = train_df["id"].tolist()
train_desc = train_df["description"].tolist()
val_vids   = val_df["id"].tolist()
val_desc   = val_df["description"].tolist()

ALL_DESCRIPTIONS = train_desc + val_desc

print(f"✅ Train videos: {len(train_vids)} | Val videos: {len(val_vids)}")
print(f"📝 Total descriptions: {len(ALL_DESCRIPTIONS)}")

# ============================================================
# 4️⃣ CLIP TEXT ENCODER (CPU, FROZEN)
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
        inputs = processor(text=chunk, return_tensors="pt",
                           padding=True, truncation=True)
        emb = clip.get_text_features(**inputs)
        emb = nn.functional.normalize(emb, dim=-1)
        all_text_embeds.append(emb.cpu())

text_embeds = torch.cat(all_text_embeds, dim=0)
NUM_TEXTS = text_embeds.size(0)

print(f"✅ Text embeddings ready: {NUM_TEXTS}")

# ============================================================
# 5️⃣ MAMBA VIDEO ENCODER (ARCHITECTURE)
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
        x = frame_embs.unsqueeze(0)
        y = self.mamba(x)
        y = self.dropout(y[:, -1])
        return nn.functional.normalize(y, dim=-1)

model = MambaVideoEncoder().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

# ============================================================
# 6️⃣ LOAD CHECKPOINT (EPOCH 20)
# ============================================================
print("⬇️ Loading epoch_20 checkpoint from S3...")

ckpt_buf = io.BytesIO()
s3.download_fileobj(
    Bucket=S3_BUCKET,
    Key=f"{CHECKPOINT_PREFIX}/epoch_{RESUME_EPOCH}.pth",
    Fileobj=ckpt_buf
)
ckpt_buf.seek(0)

checkpoint = torch.load(ckpt_buf, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])

# 🔽 FORCE NEW LEARNING RATE (ONLY ADDITION)
for param_group in optimizer.param_groups:
    param_group["lr"] = LR

print(f"✅ Resumed from epoch {checkpoint['epoch']}")

# ============================================================
# 7️⃣ UTILITIES
# ============================================================
def load_video_embeddings(vid):
    return torch.load(f"{LOCAL_EMBED_DIR}/{vid}.pt").to(DEVICE).float()

def contrastive_loss_chunked(video_embs, target_indices):
    B = video_embs.size(0)
    loss = 0.0
    for start in range(0, NUM_TEXTS, TEXT_SIM_CHUNK):
        end = min(start + TEXT_SIM_CHUNK, NUM_TEXTS)
        chunk = text_embeds[start:end].to(DEVICE)
        logits = (video_embs @ chunk.T) / TEMPERATURE
        log_probs = torch.log_softmax(logits, dim=1)
        for i in range(B):
            tgt = target_indices[i]
            if start <= tgt < end:
                loss -= log_probs[i, tgt - start]
        del chunk, logits, log_probs
    return loss / B

# ============================================================
# 8️⃣ TRAIN / VAL LOOP
# ============================================================
def run_epoch(video_ids, text_offset, train=True):
    model.train() if train else model.eval()
    total_loss, count = 0.0, 0

    for i in tqdm(range(0, len(video_ids), BATCH_SIZE), leave=False):
        batch_vids = video_ids[i:i + BATCH_SIZE]
        video_embs, target_indices = [], []

        for j, vid in enumerate(batch_vids):
            frames = load_video_embeddings(vid)
            emb = model(frames)
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
# 9️⃣ FINETUNING (EPOCHS 21 → 40)
# ============================================================
for epoch in range(RESUME_EPOCH + 1, RESUME_EPOCH + FINETUNE_EPOCHS + 1):
    train_loss = run_epoch(train_vids, text_offset=0, train=True)
    val_loss = run_epoch(val_vids, text_offset=len(train_vids), train=False)

    print(f"[Epoch {epoch}] train={train_loss:.4f} | val={val_loss:.4f}")

    buf = io.BytesIO()
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        },
        buf
    )
    buf.seek(0)

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"{CHECKPOINT_PREFIX}/epoch_{epoch}.pth",
        Body=buf.getvalue()
    )

print("🎉 FINETUNING COMPLETE — EPOCHS 21–40 SAVED")



