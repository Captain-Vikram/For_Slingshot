import cv2
import os

DATA_DIR = "hcr_data"          # Folder containing original game screenshots
OUTPUT_DIR = "hcr_data_masked" # Folder to save masked images
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Coordinates of HUD regions to black out
# Each tuple represents
BLACK_AREAS = [
    (0, 36, 288, 216),       # Top-left HUD (speed, fuel, etc.)
    (417, 588, 889, 748),    # Center-bottom HUD (buttons, ground UI)
    (59, 542, 256, 781),     # Bottom-left area (pedal, brake)
    (1043, 541, 1241, 780),  # Bottom-right area (gas pedal)
    (305, 51, 1031, 130)     # Top bar (score/time UI)
]

for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".png"):
        continue  # Skip non-image files

    img_path = os.path.join(DATA_DIR, fname)
    img = cv2.imread(img_path)

    # Apply black masks to the defined UI areas
    for x1, y1, x2, y2 in BLACK_AREAS:
        img[y1:y2, x1:x2] = 0  # Set pixels in region to black (masking out)

    # Save the masked image to the output folder
    out_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, img)
    print(f"Processed {fname}")

print("✅ All images masked successfully!")
