import sys
import os
import csv
import shutil
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()

import imagehash
from tqdm import tqdm
import cv2
import numpy as np
from itertools import combinations

MAX_HASH_DIFF = 576  # 24x24 hash = 576 bits
HASH_FILE = "hashes.txt"

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Group similar images and suggest best quality.")
    parser.add_argument("folder", help="Path to image folder")
    parser.add_argument("--threshold", type=float, default=70.0, help="Minimum similarity percentage to group images")
    return parser.parse_args()

def load_hashes_from_file():
    hashes = {}
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    filename, hashval = row
                    hashes[filename] = imagehash.hex_to_hash(hashval)
        print(f"📂 Loaded {len(hashes)} hashes from {HASH_FILE}")
    return hashes

def save_hashes_to_file(hashes):
    with open(HASH_FILE, "w", newline='') as f:
        writer = csv.writer(f)
        for filename, hashval in hashes.items():
            writer.writerow([filename, str(hashval)])

def get_image_hashes(folder):
    existing_hashes = load_hashes_from_file()
    hashes = {}
    filenames = sorted([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.heic', '.tif', '.tiff'))
    ])

    progress = tqdm(filenames, desc="Hashing images", colour="green", dynamic_ncols=True)

    for filename in progress:
        filepath = os.path.join(folder, filename)
        progress.set_description_str(f"\033[38;5;208mHashing:\033[0m {filename}")
        if filename in existing_hashes:
            hashes[filename] = existing_hashes[filename]
        else:
            try:
                img = Image.open(filepath)
                hashval = imagehash.phash(img, hash_size=24)
                hashes[filename] = hashval
            except Exception as e:
                print(f"\n⚠️ Error processing {filename}: {e}")

    save_hashes_to_file(hashes)
    return hashes

def analyze_quality(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    try:
        if ext in [".heic", ".tif", ".tiff"]:
            img = Image.open(image_path).convert("L")
            img = np.array(img)
        else:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
    except Exception as e:
        print(f"⚠️ Error reading {image_path}: {e}")
        return None

    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    high_freq = cv2.subtract(img, blurred)
    noise_std = np.std(high_freq)
    edges = cv2.Canny(img, 100, 200)
    texture_density = np.sum(edges > 0) / edges.size

    return {
        "sharpness": round(laplacian_var, 2),
        "noise": round(noise_std, 2),
        "texture": round(texture_density * 100, 2)
    }

def score_quality(q):
    return q['sharpness'] - q['noise'] + q['texture']

def group_similar_images(hashes, min_similarity):
    groups = []
    used = set()
    filenames = list(hashes.keys())

    for i, img1 in enumerate(filenames):
        if img1 in used:
            continue
        group = [img1]
        for j in range(i + 1, len(filenames)):
            img2 = filenames[j]
            if img2 in used:
                continue
            diff = hashes[img1] - hashes[img2]
            similarity = 100 - (diff / MAX_HASH_DIFF * 100)
            if similarity >= min_similarity:
                group.append(img2)
        if len(group) > 1:
            groups.append(group)
            used.update(group)
    return groups

def average_group_similarity(group, hashes):
    similarities = []
    for img1, img2 in combinations(group, 2):
        diff = hashes[img1] - hashes[img2]
        similarity = 100 - (diff / MAX_HASH_DIFF * 100)
        similarities.append(similarity)
    return round(sum(similarities) / len(similarities), 2) if similarities else 100.0

def process_groups(groups, folder, hashes):
    for idx, group in enumerate(groups, start=1):
        avg_sim = average_group_similarity(group, hashes)
        print(f"\n🧮 Group {idx} ({len(group)} images) — Avg. similarity: {avg_sim:.2f}%")
        scores = {}
        for img in group:
            q = analyze_quality(os.path.join(folder, img))
            if q:
                score = score_quality(q)
                scores[img] = score
                print(f"  📊 {img} → Sharpness: {q['sharpness']}, Noise: {q['noise']}, Texture: {q['texture']}% | Score: {score:.2f}")
        if scores:
            best = max(scores, key=scores.get)
            print(f"  ✅ Suggested to keep: {best} (Highest score in group)")

def move_duplicates(groups, folder, hashes):
    dup_dir = os.path.join(folder, "duplicates")
    os.makedirs(dup_dir, exist_ok=True)

    for group in groups:
        scores = {}
        for img in group:
            q = analyze_quality(os.path.join(folder, img))
            if q:
                scores[img] = score_quality(q)
        if scores:
            best = max(scores, key=scores.get)
            for img in group:
                if img != best:
                    src = os.path.join(folder, img)
                    dst = os.path.join(dup_dir, img)
                    try:
                        shutil.move(src, dst)
                        print(f"📦 Moved duplicate: {img} → duplicates/")
                    except Exception as e:
                        print(f"⚠️ Could not move {img}: {e}")

def main():
    args = parse_args()
    if not os.path.isdir(args.folder):
        print(f"Error: '{args.folder}' is not a valid directory.")
        sys.exit(1)

    hashes = get_image_hashes(args.folder)
    groups = group_similar_images(hashes, min_similarity=args.threshold)
    process_groups(groups, args.folder, hashes)

    print("\n📘 Quality Metrics Explanation:")
    print("  🔹 Sharpness: Variance of Laplacian (typical range: 0–1000+)")
    print("     → Higher is better. Indicates edge clarity and focus.")
    print("  🔹 Noise: Std. dev. of high-frequency content (typical range: 0–50)")
    print("     → Lower is better. High values suggest grain or compression artifacts.")
    print("  🔹 Texture Detail: Edge pixel density (0–100%)")
    print("     → Higher is better. Reflects fine surface detail and structure.")
    print("  ✅ Suggested image is based on: sharpness - noise + texture")

    move_duplicates(groups, args.folder, hashes)

if __name__ == "__main__":
    main()
