# ============================================================
# 🎯 VIDEO-LEVEL DESCRIPTION TRAINING (10 EPOCHS)
# MAMBA TEMPORAL SUMMARY + CLIP SHARED SPACE
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
LR = 1e-4
DROPOUT = 0.2
BATCH_SIZE = 4
EPOCHS = 10
TEMPERATURE = 0.1

TEXT_ENCODE_CHUNK = 256                                       # text descriptionsare process in chunks of 256 descriptions
TEXT_SIM_CHUNK = 512                                          # video embeddings are processed in chunks of 512 for similarity computation

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
# 2️⃣ SYNC ALL FRAME EMBEDDINGS FROM S3 (WITH x / y COUNTS)
# ============================================================
print("⬇️ Syncing embeddings from S3...")

total_s3_embeddings = 0
downloaded_embeddings = 0

continuation = None
while True:
    kwargs = dict(Bucket=S3_BUCKET, Prefix=EMBED_PREFIX)
    if continuation:
        kwargs["ContinuationToken"] = continuation                  # continue to next page in bucket if next page has files 

    resp = s3.list_objects_v2(**kwargs)
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".pt"):                                      # only .pt files are embeddings
            total_s3_embeddings += 1

            fname = key.split("/")[-1]                               # get filename from key
            local_path = f"{LOCAL_EMBED_DIR}/{fname}"

            if not os.path.exists(local_path):                       # we download only if it is missing
                s3.download_file(S3_BUCKET, key, local_path)         
                downloaded_embeddings += 1

    if resp.get("IsTruncated"):
        continuation = resp["NextContinuationToken"]
    else:
        break

EMBED_VIDS = set(f.replace(".pt", "") for f in os.listdir(LOCAL_EMBED_DIR))  # stored only the id's of the video whose embeddings are there

print(
    f"📦 Embeddings loaded: {len(EMBED_VIDS)} / {total_s3_embeddings} "
    f"(downloaded this run: {downloaded_embeddings})"
)

# ============================================================
# 3️⃣ LOAD CSVs (FILTERED)
# ============================================================
train_df = pd.read_csv(TRAIN_CSV).rename(columns={"descriptions": "description"})
val_df   = pd.read_csv(VAL_CSV).rename(columns={"descriptions": "description"})

train_df = train_df[train_df["id"].isin(EMBED_VIDS)]        # select only those trainning videos for which embeddings are present
val_df   = val_df[val_df["id"].isin(EMBED_VIDS)]            # select only those val videos for which embeddings are present to ensure no

train_vids = train_df["id"].tolist()                      #["00HFP","00IQ3",.......7500 videos]
train_desc = train_df["description"].tolist()             #["A person is sitting on a table","A person is walking","A lady is moving".....7500 descriptions]

val_vids   = val_df["id"].tolist()                       #["AX123","ax124",.......1500 videos]
val_desc   = val_df["description"].tolist()              #["a man is standing","a lady is sitting ".....1500 descriptions]          

ALL_DESCRIPTIONS = train_desc + val_desc                 #["A person is sitting on a table","A girl is siting" , .....9000 descriptions]

print(f"✅ Train videos: {len(train_vids)} | Val videos: {len(val_vids)}")  # length of train videos and val videos
print(f"📝 Total descriptions: {len(ALL_DESCRIPTIONS)}")                   # length of all_descriptions

# ============================================================
# 4️⃣ CLIP TEXT ENCODER (CPU, CHUNKED)
# ============================================================
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

clip.eval()
for p in clip.parameters(): 
    p.requires_grad = False                                 # freeze the clip model weights are not updated 

all_text_embeds = []                                        #list to store text embeddings

with torch.no_grad():
    for i in tqdm(range(0, len(ALL_DESCRIPTIONS), TEXT_ENCODE_CHUNK), # loop in group of 256-256 till we dont reach the end
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
        all_text_embeds.append(emb.cpu())                  # we brought these text embeddings to cpu because gpu was running out of memory

text_embeds = torch.cat(all_text_embeds, dim=0)
NUM_TEXTS = text_embeds.size(0)

print(f"✅ Text embeddings ready: {NUM_TEXTS}")

# ============================================================
# 5️⃣ MAMBA VIDEO ENCODER (TRUE TEMPORAL SUMMARY)
# ============================================================
class MambaVideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = Mamba(
            d_model=EMBED_DIM,
            d_state=16,                                 # 16 dimensional state                  
            d_conv=4,                                   # each frame can directly see information from next 4 frames 
            expand=2                                    # neural nets are doubled to capture more features         
        )
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, frame_embs):
        x = frame_embs.unsqueeze(0)                      # (1, T, 768) converted to the standard form clip of 1 frame only
        y = self.mamba(x)
        y = self.dropout(y[:, -1])                       # full video summary
        return nn.functional.normalize(y, dim=-1)        # normalize the final video embedding

model = MambaVideoEncoder().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# ============================================================
# 6️⃣ UTILITIES
# ============================================================
def load_video_embeddings(vid):
    return torch.load(f"{LOCAL_EMBED_DIR}/{vid}.pt").to(DEVICE).float() 

def contrastive_loss_chunked(video_embs, target_indices):
    B = video_embs.size(0)
    loss = 0.0
    
    for start in range(0, NUM_TEXTS, TEXT_SIM_CHUNK):        # process text embeddings in chunks of 512 for similarity computation
        end = min(start + TEXT_SIM_CHUNK, NUM_TEXTS)
        chunk = text_embeds[start:end].to(DEVICE)

        logits = (video_embs @ chunk.T) / TEMPERATURE
        log_probs = torch.log_softmax(logits, dim=1)

        for i in range(B):
            tgt = target_indices[i]
            if start <= tgt < end:
                loss -= log_probs[i, tgt - start]   # because log_probs are large negative and best to 0 

        del chunk, logits, log_probs                # free them from memory

    return loss / B

# ============================================================
# 7️⃣ TRAIN / VAL LOOP (NO FRAME MIXING)
# ============================================================
def run_epoch(video_ids, text_offset, train=True):
    model.train() if train else model.eval()
    total_loss, count = 0.0, 0

    for i in tqdm(range(0, len(video_ids), BATCH_SIZE), leave=False):   # leave=false when progress bar finishes, remove it 
        batch_vids = video_ids[i:i + BATCH_SIZE]

        video_embs = []                                       # list that stores video embeddings for the batch
        target_indices = []                                   # list that stores target text indices for the batch

        for j, vid in enumerate(batch_vids):                  # j is index within the batch, vid is video id
            frames = load_video_embeddings(vid)               # i is the starting video index of that batch 
            emb = model(frames)
            video_embs.append(emb)
            target_indices.append(text_offset + i + j)        # durining training , text-offset is 0 , during val it is len(train_vids) to move to start index of val descriptions    

        video_embs = torch.cat(video_embs, dim=0)             # 4,768
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
# 8️⃣ TRAINING (10 EPOCHS)


# ============================================================
for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_vids, text_offset=0, train=True)
    val_loss   = run_epoch(val_vids, text_offset=len(train_vids), train=False)

    print(f"[Epoch {epoch}] train={train_loss:.4f} | val={val_loss:.4f}")

    buf = io.BytesIO()                                   # checkpoint saved to s3 without writing anything to disk
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

print("🎉 TRAINING COMPLETE — VIDEO-LEVEL MODEL READY")