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
S3_META_PREFIX = "charades_data"
EMBED_PREFIX = "charades_clip_embeddings_final"
CHECKPOINT_PREFIX = "mamba_charades_retraining"

LOCAL_META_DIR = "./charades_data"
LOCAL_EMBED_DIR = "/tmp/charades_embeddings"
os.makedirs(LOCAL_META_DIR, exist_ok=True)
os.makedirs(LOCAL_EMBED_DIR, exist_ok=True)

CLASSES_FILE = f"{LOCAL_META_DIR}/Charades_v1_classes.txt"
TRAIN_CSV = f"{LOCAL_META_DIR}/Charades_v1_train.csv"
VAL_CSV = f"{LOCAL_META_DIR}/Charades_v1_test.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM = 768
LR = 1e-4
DROPOUT = 0.2
BATCH_SIZE = 4
MAX_EPOCHS = 40
EARLY_STOPPING = 7
RESUME_EPOCH = 26

s3 = boto3.client("s3")

# ============================================================
# 1️⃣ DOWNLOAD METADATA
# ============================================================
def download_if_missing(key, local_path):
    if not os.path.exists(local_path):
        s3.download_file(S3_BUCKET, key, local_path)

download_if_missing(f"{S3_META_PREFIX}/Charades_v1_classes.txt", CLASSES_FILE)
download_if_missing(f"{S3_META_PREFIX}/Charades_v1_train.csv", TRAIN_CSV)
download_if_missing(f"{S3_META_PREFIX}/Charades_v1_test.csv", VAL_CSV)

# ============================================================
# 2️⃣ SYNC ALL EMBEDDINGS (PAGINATED)
# ============================================================
print("⬇️ Syncing all embeddings from S3...")

continuation = None
while True:
    kwargs = dict(Bucket=S3_BUCKET, Prefix=EMBED_PREFIX)
    if continuation:
        kwargs["ContinuationToken"] = continuation

    resp = s3.list_objects_v2(**kwargs)                               # call one page of directory 
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        if not key.endswith(".pt"):
            continue
        fname = key.split("/")[-1]                                    # extract only the file names 
        local_path = f"{LOCAL_EMBED_DIR}/{fname}"
        if not os.path.exists(local_path):
            s3.download_file(S3_BUCKET, key, local_path)

    if resp.get("IsTruncated"):                                        #   if more files after this page
        continuation = resp["NextContinuationToken"]
    else:
        break

EMBED_VIDS = set(f.replace(".pt", "") for f in os.listdir(LOCAL_EMBED_DIR))   # embed_vids make a set of video id e.g "1bc23" 
print(f"✅ Total embeddings available: {len(EMBED_VIDS)}")
 
# ============================================================
# 3️⃣ LOAD LABELS (CSV ∩ EMBEDDINGS)
# ============================================================
class_id_to_idx = {}                                                   # c001 :0
class_texts = []                                                       # empty list to show text description of all classes 

with open(CLASSES_FILE) as f:                                          # labels file 
    for idx, line in enumerate(f):
        cid, text = line.strip().split(" ", 1)                             
        class_id_to_idx[cid] = idx                                     #  c123 :0  
        class_texts.append(f"{cid} {text}")                            # "C123 opening refrigerator"


NUM_CLASSES = len(class_texts)

def load_multilabels(csv_path):                                      
    df = pd.read_csv(csv_path)
    out = {}                                                            # dictionary to store video_id → {label1, label2, ...}
    for _, r in df.iterrows():
        if pd.isna(r["actions"]):                                       # if no labels for a video skip it 
            continue
        vid = r["id"]                                                   # if action found we extract this video id
        if vid not in EMBED_VIDS:                                       #  if not in the videos whose embeddings we have calculated
            continue
        for act in r["actions"].split(";"):                             # same videos multiple 
            lbl = act.split(" ")[0]                                     # video_id → {label1, label2, ...
            out.setdefault(vid, set()).add(lbl)                         
    return out

train_labels = load_multilabels(TRAIN_CSV)                              # extract usable training video IDs
val_labels = load_multilabels(VAL_CSV)                                  # extract usable testing video IDs

TRAIN_VIDS = list(train_labels.keys())                                  # TRAIN_VIDS = [ "1A2B3C", "4D5E6F",...]
VAL_VIDS = list(val_labels.keys())                                      # VAL_VIDS =["0X654D,"123Ax"]

print(f"✅ Train videos used: {len(TRAIN_VIDS)}")
print(f"✅ Val videos used:   {len(VAL_VIDS)}")

# ============================================================
# 4️⃣ CLIP TEXT ENCODER (FROZEN)
# ============================================================
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
# CLIPProcessor handles text tokenization, padding, and input formattin
clip.eval()                                                             # clip model into evaluation mode
for p in clip.parameters():                                             # parameters of the clip model dont need to be updated
    p.requires_grad = False                                             #  no dropout ,no randomness,consistent embeddings

with torch.no_grad():                                                   # disable gradient tracking for clip
    # convert all class labels  "C123 opening refrigerator" to tokenized tensors shared space 
    text_inputs = processor(text=class_texts, return_tensors="pt", padding=True)  
    text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}     # move tensors to gpu
    text_embeds = clip.get_text_features(**text_inputs)                 # embeddings for all classes like "C123 opening refrigerator
    # Encode all action labels into CLIP’s shared image–text embedding space (768-D)
    text_embeds = nn.functional.normalize(text_embeds, dim=-1).float()  #  Normalize text embeddings so similarity is cosine similarity

# ============================================================
# 5️⃣ MAMBA VIDEO ENCODER
# ============================================================
class MambaVideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = Mamba(d_model=EMBED_DIM, d_state=16, d_conv=4, expand=2)    #internal memory size,local mixing, expand
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, frame_emb):                                                # mamba receives one frame embedding
        x = frame_emb.unsqueeze(0).unsqueeze(1)                                  # single frame passed as batch of 1 frame
        y = self.mamba(x)
        y = self.dropout(y[:, -1])                                               #  (1, 1, 768) --> 1,768  --> this is the returned embeddings y 
        return nn.functional.normalize(y, dim=-1)                                # Normalize video embedding to stay compatible with cosine similarity in shared space

model = MambaVideoEncoder().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# ============================================================
# 6️⃣ RESUME FROM EPOCH 26
# ============================================================
ckpt_key = f"{CHECKPOINT_PREFIX}/epoch_{RESUME_EPOCH}.pth"
ckpt_local = f"/tmp/epoch_{RESUME_EPOCH}.pth"        
s3.download_file(S3_BUCKET, ckpt_key, ckpt_local)                                 # download checkpoint from s3 to local 

ckpt = torch.load(ckpt_local, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])                                        #  restore mamba model weights
optimizer.load_state_dict(ckpt["optimizer_state"])                                # load the optimizer state
best_val = ckpt.get("best_val", float("inf"))                                     # restore previous best_val and patience
patience = ckpt.get("patience", 0)

start_epoch = RESUME_EPOCH + 1
print(f"✅ Resumed from epoch {RESUME_EPOCH}")

# ============================================================
# 7️⃣ LR SCHEDULER (NEW)
# ============================================================
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2, verbose=True                    # loss should be minimum/decreasing , if not then reduced learning rate by half , show logs 
)

# ============================================================
# 8️⃣ LOSS
# ============================================================
def contrastive_loss(logits, targets):                                             #logits tell similarity with each action label, target tells correct labels [0,0,0,1,....]
    logits = logits / 0.1                                                          #to make the labels standout 0.1/0.1 -> 1
    log_probs = torch.log_softmax(logits, dim=1)
    loss = -(targets * log_probs).sum(dim=1) / (targets.sum(dim=1) + 1e-6)
    return loss.mean()

def load_video_embeddings(vid):
    return torch.load(f"{LOCAL_EMBED_DIR}/{vid}.pt").to(DEVICE).float()

def run_epoch(video_ids, labels, train=True):
    model.train() if train else model.eval()                                        # training enables dropout and gradient flow , eval - weights constnt
    total_loss, count = 0.0, 0 

    if train: 
        optimizer.zero_grad()                                                        # clrs prev grads

    for i in tqdm(range(0, len(video_ids), BATCH_SIZE), leave=False):                # iterates in btch of 4-4
        batch = video_ids[i:i+BATCH_SIZE]
        video_embs, targets = [], []                                                 # final embedding per video  , multi-hot label vector per video target=[0,0,1,1,0.....] as per the model
 
        for vid in batch:                                                            # one video in batch 
            frames = load_video_embeddings(vid)                                      # Load precomputed CLIP image embeddings 
            for f in frames:
                emb = model(f)                                                       #  Aggregate frame-level CLIP embeddings over time using Mamba to produce a single video embedding in the same shared space
            video_embs.append(emb)                                                   # final-vid embedng summarise all frames  (batch_size,768)

            t = torch.zeros(NUM_CLASSES, device=DEVICE)
            for lbl in labels[vid]:
                t[class_id_to_idx[lbl]] = 1.0                                        # mark ground-truth label as 1 
            targets.append(t)                                                        # stores the vector [1,0,0,0,1]

        video_embs = torch.cat(video_embs)                                           #Stacks video embeddings into a batch tens
        targets = torch.stack(targets)
        logits = video_embs @ text_embeds.T      # Compute cosine similarity between video embeddings and all text label embeddings in the shared CLIP space
        loss = contrastive_loss(logits, targets) # Train the model by pulling video embeddings closer to correct text embeddings and pushing away others in the shared space

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * video_embs.size(0)                                # average batch loss*number of videos 
        count += video_embs.size(0)                                                   # count becomes equal to batch size 

    return total_loss / count

# ============================================================
# 9️⃣ TRAINING LOOP (EPOCH 27+)
# ============================================================
for epoch in range(start_epoch, MAX_EPOCHS + 1):
    train_loss = run_epoch(TRAIN_VIDS, train_labels, True)
    val_loss = run_epoch(VAL_VIDS, val_labels, False)

    print(f"Epoch {epoch} | train={train_loss:.4f} | val={val_loss:.4f}")
    scheduler.step(val_loss)

    buf = io.BytesIO()        # chkpnt to s3
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val": best_val,
        "patience": patience
    }, buf)
    buf.seek(0)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"{CHECKPOINT_PREFIX}/epoch_{epoch}.pth",
        Body=buf.getvalue()
    )

    if val_loss < best_val:
        best_val = val_loss
        patience = 0
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=f"{CHECKPOINT_PREFIX}/best_model.pth",
            Body=buf.getvalue()
        )
        print("✅ Best model updated")
    else:
        patience += 1

    if patience >= EARLY_STOPPING:
        print("⏹ Early stopping")
        break