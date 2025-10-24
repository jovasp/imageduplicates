## 📸 Image Similarity Clustering & Quality Analysis

This Python script groups similar images using perceptual hashing and recommends the best-quality image to keep. It supports advanced formats like HEIC and TIFF, computes sharpness/noise/texture metrics, and automatically moves lower-quality duplicates to a subdirectory.

---

### 🔧 Features

- ✅ Perceptual hashing with `hash_size=24` (576-bit precision)
- ✅ Supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.heic`, `.tif`, `.tiff`
- ✅ Groups images with similarity above a threshold (default: 70%)
- ✅ Computes image quality metrics:
  - **Sharpness**: Laplacian variance
  - **Noise**: High-frequency standard deviation
  - **Texture**: Edge density
- ✅ Suggests best image per group
- ✅ Moves duplicates to `duplicates/` subdirectory
- ✅ Caches hashes in `hashes.txt` for faster re-runs
- ✅ Progress bars via `tqdm`

---

### 🚀 Installation

```bash
pip install pillow pillow-heif imagehash opencv-python tqdm numpy
```

---

### 📁 Usage

```bash
python duplicateimagefinder.py /path/to/images --threshold 85
```

- `--threshold`: Minimum similarity percentage to group images (default: 70)

---

### 📂 Output

- Prints similarity and quality metrics per group
- Keeps highest-scoring image in place
- Moves other images to `/path/to/images/duplicates/`

---

### 📘 Quality Metrics Explained

| Metric     | Description                                      | Ideal Value |
|------------|--------------------------------------------------|-------------|
| Sharpness  | Variance of Laplacian (edge clarity)             | High        |
| Noise      | Std. dev. of high-frequency content              | Low         |
| Texture    | Edge pixel density (surface detail)              | High        |

Score formula: `sharpness - noise + texture`

---

### 🧠 Why Use This?

- Clean up photo libraries
- Remove low-quality duplicates
- Automate image selection for archival or publishing
- Works offline, customizable, and scriptable

---

### 📄 License

MIT License. See `LICENSE` file for details.

---
