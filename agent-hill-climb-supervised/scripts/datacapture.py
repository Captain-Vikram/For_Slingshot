# Import necessary modules
import mss
import numpy as np
import cv2
import keyboard
import os
import time
import pygetwindow as gw
import re

# Detect the "Hill Climb Racing" window and retrieve its position and size
win = gw.getWindowsWithTitle('Hill Climb Racing')[0]
print("Window position:", win.left, win.top)
print("Window size:", win.width, win.height)

# Set u the output folder for saving captured frames
SAVE_DIR = "hcr_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# Check for existing frames to continue numbering from the last saved one
existing_frames = [f for f in os.listdir(SAVE_DIR) if f.startswith("frame_") and f.endswith(".png")]

if existing_frames:
    # Extract numeric part from filenames and find the highest frame number
    last_num = max(
        int(re.search(r"frame_(\d+)_", f).group(1))
        for f in existing_frames
        if re.search(r"frame_(\d+)_", f)
    )
    frame_id = last_num + 1
else:
    frame_id = 0

print(f"Starting from frame ID {frame_id}")
print("Starting data capture. Press ESC to stop.")

# Begin screen capture and input recording loop
with mss.mss() as sct:
    # Define capture area to match the game window
    monitor = {"top": win.top, "left": win.left, "width": win.width, "height": win.height}

    while True:
        # Stop capture when ESC is pressed
        if keyboard.is_pressed("esc"):
            print("Stopping capture.")
            break

        # Determine current player input
        if keyboard.is_pressed("LEFT"):
            action = "BRAKE"
        elif keyboard.is_pressed("RIGHT"):
            action = "ACCEL"
        else:
            action = "NONE"

        # Take a screenshot of the game window
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR (remove alpha channel)

        # Save frame with timestamped action label
        filename = f"{SAVE_DIR}/frame_{frame_id:05d}_{action}.png"
        cv2.imwrite(filename, img)

        print(f"Saved {filename}")
        frame_id += 1

        # Capture rate control (5 frames per second)
        time.sleep(0.2)
