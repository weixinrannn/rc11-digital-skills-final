import shutil
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm
import imagehash

import torch
import open_clip


# =====================================
# 1. Path Configuration
# =====================================
RAW_DIR = Path(r"D:\00\raw_paintings")
FINAL_DIR = Path(r"D:\00\paintings_final_selected")
REMOVED_DIR = Path(r"D:\00\paintings_removed_review")
CSV_PATH = Path(r"D:\00\painting_metadata_final.csv")

FINAL_DIR.mkdir(parents=True, exist_ok=True)
REMOVED_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# =====================================
# 2. Parameters
# =====================================
PHASH_THRESHOLD = 6
POS_THRESHOLD = 0.24
MARGIN_THRESHOLD = 0.02


# =====================================
# 3. Utility Functions
# =====================================
def file_md5(path: Path, chunk_size=8192):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def safe_open_image(path: Path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def image_quality_score(path: Path):
    img = safe_open_image(path)
    if img is None:
        return 0
    width, height = img.size
    pixel_score = width * height
    file_score = path.stat().st_size
    return pixel_score + file_score


def choose_better_image(path1: Path, path2: Path):
    score1 = image_quality_score(path1)
    score2 = image_quality_score(path2)
    return path1 if score1 >= score2 else path2


# =====================================
# 4. Collect Images
# =====================================
image_files = [p for p in RAW_DIR.iterdir() if p.suffix.lower() in VALID_EXTS]

if not image_files:
    print("No images found in raw_paintings folder.")
    raise SystemExit

print(f"Original image count: {len(image_files)}")


# =====================================
# 5. Step 1: Remove Exact Duplicates (MD5)
# =====================================
md5_map = {}
exact_dup_log = []

for img_path in tqdm(image_files, desc="Step 1/3 Exact duplicate check"):
    try:
        md5 = file_md5(img_path)

        if md5 not in md5_map:
            md5_map[md5] = img_path
        else:
            old_path = md5_map[md5]
            keep = choose_better_image(old_path, img_path)
            drop = img_path if keep == old_path else old_path
            md5_map[md5] = keep

            exact_dup_log.append({
                "image_name": drop.name,
                "image_path": str(drop),
                "status": "removed_exact_duplicate",
                "reason": "same_md5",
                "best_positive_prompt": "",
                "best_positive_score": None,
                "best_negative_prompt": "",
                "best_negative_score": None,
                "margin": None
            })

    except Exception as e:
        exact_dup_log.append({
            "image_name": img_path.name,
            "image_path": str(img_path),
            "status": "error",
            "reason": f"md5_error:{e}",
            "best_positive_prompt": "",
            "best_positive_score": None,
            "best_negative_prompt": "",
            "best_negative_score": None,
            "margin": None
        })

stage1_files = list(md5_map.values())
print(f"After exact duplicate removal: {len(stage1_files)}")


# =====================================
# 6. Step 2: Remove Near Duplicates (pHash)
# =====================================
kept_files = []
kept_hashes = []
similar_dup_log = []

for img_path in tqdm(stage1_files, desc="Step 2/3 Similar image check"):
    img = safe_open_image(img_path)

    if img is None:
        similar_dup_log.append({
            "image_name": img_path.name,
            "image_path": str(img_path),
            "status": "error",
            "reason": "open_error",
            "best_positive_prompt": "",
            "best_positive_score": None,
            "best_negative_prompt": "",
            "best_negative_score": None,
            "margin": None
        })
        continue

    try:
        current_hash = imagehash.phash(img)
    except Exception as e:
        similar_dup_log.append({
            "image_name": img_path.name,
            "image_path": str(img_path),
            "status": "error",
            "reason": f"phash_error:{e}",
            "best_positive_prompt": "",
            "best_positive_score": None,
            "best_negative_prompt": "",
            "best_negative_score": None,
            "margin": None
        })
        continue

    found_similar = False

    for i, kept_path in enumerate(kept_files):
        dist = current_hash - kept_hashes[i]

        if dist <= PHASH_THRESHOLD:
            better = choose_better_image(kept_path, img_path)
            worse = img_path if better == kept_path else kept_path

            if better == img_path:
                kept_files[i] = img_path
                kept_hashes[i] = current_hash

            similar_dup_log.append({
                "image_name": worse.name,
                "image_path": str(worse),
                "status": "removed_similar_duplicate",
                "reason": f"phash_distance={dist}",
                "best_positive_prompt": "",
                "best_positive_score": None,
                "best_negative_prompt": "",
                "best_negative_score": None,
                "margin": None
            })

            found_similar = True
            break

    if not found_similar:
        kept_files.append(img_path)
        kept_hashes.append(current_hash)

deduped_files = kept_files
print(f"After similar duplicate removal: {len(deduped_files)}")


# =====================================
# 7. Step 3: Semantic Filtering (CLIP)
# =====================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device)
model.eval()

positive_prompts = [
    "a painting of an interior room with a window",
    "a painting of a domestic interior with sunlight entering through a window",
    "a painting of a person inside looking out through a window",
    "a painting of an interior space with a visible window and outside view",
    "a painting of a quiet room with a window",
    "a painting of an indoor scene with strong window light",
]

negative_prompts = [
    "a painting of an outdoor landscape",
    "a painting without a window",
    "a portrait painting with no relation to a window",
    "an abstract painting",
    "a painting of only outdoor scenery",
    "a blurry low quality image",
]

all_prompts = positive_prompts + negative_prompts

with torch.no_grad():
    text_tokens = tokenizer(all_prompts).to(device)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

selection_results = []

for img_path in tqdm(deduped_files, desc="Step 3/3 Semantic filtering"):
    img = safe_open_image(img_path)

    if img is None:
        continue

    try:
        image_input = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
            similarity = similarity.cpu().numpy()

        pos_scores = similarity[:len(positive_prompts)]
        neg_scores = similarity[len(positive_prompts):]

        best_pos_score = float(pos_scores.max())
        best_neg_score = float(neg_scores.max())
        margin = best_pos_score - best_neg_score

        keep = (best_pos_score >= POS_THRESHOLD) and (margin >= MARGIN_THRESHOLD)

        if keep:
            shutil.copy2(img_path, FINAL_DIR / img_path.name)
        else:
            shutil.copy2(img_path, REMOVED_DIR / img_path.name)

    except Exception:
        continue


# =====================================
# 8. Final Summary
# =====================================
selected_count = len(list(FINAL_DIR.iterdir()))
removed_count = len(list(REMOVED_DIR.iterdir()))

print("\n===== PROCESS COMPLETED =====")
print(f"Original images: {len(image_files)}")
print(f"Final selected images: {selected_count}")
print(f"Removed/review images: {removed_count}")
print(f"Final folder: {FINAL_DIR}")
print(f"Removed folder: {REMOVED_DIR}")
print(f"Metadata CSV: {CSV_PATH}")
