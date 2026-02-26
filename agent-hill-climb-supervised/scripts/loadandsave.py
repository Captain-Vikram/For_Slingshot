import cv2
import numpy as np
import os

# This script processes masked game screenshots, resizes them,
# assigns control-action labels (ACCEL, BRAKE, NONE),
# normalizes pixel values, and saves them as a NumPy dataset (.npz).

# SETTINGS
IMG_SIZE = 64  # target image size (64x64 pixels)
DATA_DIR = "hcr_data_masked"  # folder containing masked images
OUTPUT_FILE = "./data/hcr_dataset.npz"  # output dataset file

# DATA LOADING
images, labels = [], []

for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".png"):
        continue  # skip non-image files

    img_path = os.path.join(DATA_DIR, fname)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # resize to fixed dimensions
    images.append(img)

    # Assign labels based on filename
    # 0 = ACCEL, 1 = BRAKE, 2 = NONE
    if "ACCEL" in fname:
        labels.append(0)
    elif "BRAKE" in fname:
        labels.append(1)
    else:
        labels.append(2)

# CONVERT TO NUMPY ARRAYS
X = np.array(images, dtype=np.float32) / 255.0  # normalize pixel values to [0, 1]
y = np.array(labels, dtype=np.int32)

print(f"✅ Dataset loaded: {X.shape[0]} images")
print(f"Image shape: {X.shape[1:]} | Labels shape: {y.shape}")

# SAVE TO DISK
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)  # ensure output folder exists
np.savez(OUTPUT_FILE, X=X, y=y)
print(f"💾 Dataset saved to {OUTPUT_FILE}")
