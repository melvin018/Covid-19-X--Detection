"""
generate_sample_data.py

Creates a small synthetic dataset of "chest X-ray-like" grayscale images
across three classes: COVID, NonCOVID, Normal.

WHY THIS EXISTS:
The real COVID-19 Radiography Database (the dataset this project type is
normally trained on) is not available in this environment. This script
generates structurally similar synthetic images (same folder layout, same
file types, same image stats) so the full pipeline -- loading, preprocessing,
training, evaluation -- can be built, run, and verified end-to-end.

TO USE YOUR REAL DATA:
Replace the contents of data/raw/<ClassName>/ with your actual X-ray images
(jpg/png) and skip running this script. The rest of the pipeline (preprocess.py,
train.py, evaluate.py) works unchanged on real data because it only assumes
the standard ImageFolder-style layout:

    data/raw/COVID/*.png
    data/raw/NonCOVID/*.png
    data/raw/Normal/*.png

The popular public dataset with this exact structure is the COVID-19
Radiography Database on Kaggle (Tawsifur Rahman et al.), commonly used for
this exact project type.
"""

import os
import numpy as np
import cv2

CLASSES = ["COVID", "NonCOVID", "Normal"]
IMAGES_PER_CLASS = 40  # small, fast sample set
IMG_SIZE = 256
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")


def make_synthetic_xray(class_name, rng):
    """
    Builds a grayscale image with class-dependent texture statistics so a
    CNN has *something* learnable to distinguish, without using any real
    patient data. This is intentionally simple -- it exists to validate the
    pipeline, not to simulate real radiological findings.
    """
    base = rng.normal(loc=120, scale=20, size=(IMG_SIZE, IMG_SIZE))

    # Vignette to mimic chest cavity falloff toward edges
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    center = IMG_SIZE / 2
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    vignette = 1 - (dist / dist.max()) * 0.5
    img = base * vignette

    # Class-specific synthetic texture cues (purely for pipeline variety)
    if class_name == "COVID":
        # Patchy bright "opacity" blobs, common ground-glass-opacity analogue
        for _ in range(rng.integers(4, 9)):
            cx, cy = rng.integers(40, IMG_SIZE - 40, size=2)
            r = rng.integers(15, 35)
            yy2, xx2 = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
            mask = (xx2 - cx) ** 2 + (yy2 - cy) ** 2 < r ** 2
            img[mask] += rng.normal(40, 10)
    elif class_name == "NonCOVID":
        # Fewer, larger, lower-contrast patches
        for _ in range(rng.integers(1, 4)):
            cx, cy = rng.integers(40, IMG_SIZE - 40, size=2)
            r = rng.integers(25, 50)
            yy2, xx2 = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
            mask = (xx2 - cx) ** 2 + (yy2 - cy) ** 2 < r ** 2
            img[mask] += rng.normal(20, 8)
    # Normal: no added opacity patches, just the base + vignette

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def main():
    rng = np.random.default_rng(seed=42)
    os.makedirs(OUT_DIR, exist_ok=True)

    for class_name in CLASSES:
        class_dir = os.path.join(OUT_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(IMAGES_PER_CLASS):
            img = make_synthetic_xray(class_name, rng)
            path = os.path.join(class_dir, f"{class_name.lower()}_{i:03d}.png")
            cv2.imwrite(path, img)
        print(f"Generated {IMAGES_PER_CLASS} sample images for class '{class_name}' -> {class_dir}")

    print("\nDone. This is synthetic sample data for pipeline validation only.")
    print("Swap in real X-ray images (same folder structure) for a real model.")


if __name__ == "__main__":
    main()
