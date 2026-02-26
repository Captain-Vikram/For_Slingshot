# overlay_hcr.py
import sys
import time
import threading
from collections import deque

import numpy as np
import cv2
import pyautogui
import pygetwindow as gw
from pynput.keyboard import Controller, Key
from tensorflow.keras.models import load_model

from PyQt5 import QtCore, QtGui, QtWidgets

# Setting up
MODEL_PATH = "./model/best_model.h5"
IMG_SIZE = 64
GAME_TITLE = "Hill Climb Racing"  # window title substring

BLACK_AREAS = [
    (0, 40, 269, 206),
    (391, 550, 833, 697),
    (50,500,242,729),
    (989,500,1178,729)
]

ACCEL_KEY = Key.right
BRAKE_KEY = Key.left

# Inference timing
SLEEP_BETWEEN_FRAMES = 0.05 

# Shared state (threadsafeish)
state = {
    "fps": 0.0,
    "action": "NONE",
    "confidence": 0.0,
    "running": True,
    "game_rect": None  # (left, top, width, height)
}
state_lock = threading.Lock()

# Utility functions (same logic as your script)
def find_game_window():
    wins = gw.getWindowsWithTitle(GAME_TITLE)
    if not wins:
        return None
    # pick the first matching window
    w = wins[0]
    return (w.left, w.top, w.width, w.height)

def mask_ui(frame):
    masked = frame.copy()
    for (x1, y1, x2, y2) in BLACK_AREAS:
        cv2.rectangle(masked, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return masked

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = mask_ui(frame)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# Inference thread
def inference_loop():
    # load model and keyboard in background thread
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print("Failed to load model:", e)
        state["running"] = False
        return

    keyboard = Controller()

    prev_time = time.time()
    fps_deque = deque(maxlen=30)

    # try to get initial game rect
    rect = find_game_window()
    if rect is None:
        print("Could not find game window. Please ensure Hill Climb Racing is running and the title matches.")
        state["running"] = False
        return

    while state["running"]:
        rect = find_game_window()
        if rect is None:
            # game closed or not found; keep trying briefly
            time.sleep(0.5)
            continue

        left, top, width, height = rect
        # update shared game_rect
        with state_lock:
            state["game_rect"] = rect

        # 1. Capturee
        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        frame = np.array(screenshot)

        # 2. Preprocess
        try:
            input_frame = preprocess_frame(frame)
        except Exception as e:
            print("Preprocess error:", e)
            time.sleep(SLEEP_BETWEEN_FRAMES)
            continue

        # 3. Predict
        try:
            pred = model.predict(input_frame, verbose=0)[0]
            action_index = int(np.argmax(pred))
            confidence = float(pred[action_index])
            action = ["ACCEL", "BRAKE", "NONE"][action_index]
        except Exception as e:
            print("Prediction error:", e)
            time.sleep(SLEEP_BETWEEN_FRAMES)
            continue

        # 4. Press keys
        # release first
        keyboard.release(ACCEL_KEY)
        keyboard.release(BRAKE_KEY)
        THRESH = 0.6
        if confidence >= THRESH:
            if action == "ACCEL":
                keyboard.press(ACCEL_KEY)
            elif action == "BRAKE":
                keyboard.press(BRAKE_KEY)
        # else none pressed

        # 5. FPS
        curr_time = time.time()
        loop_dt = curr_time - prev_time if prev_time else 0.0
        prev_time = curr_time
        fps = 1.0 / loop_dt if loop_dt > 0 else 0.0
        fps_deque.append(fps)
        avg_fps = sum(fps_deque) / len(fps_deque)

        # 6. update shared state
        with state_lock:
            state["fps"] = avg_fps
            state["action"] = action
            state["confidence"] = confidence

        # keep history of actions
        history = state.get("history", [])
        history.append(action)
        if len(history) > 20:  # max history length
            history.pop(0)

        # store probabilities as well
        with state_lock:
            state["fps"] = avg_fps
            state["action"] = action
            state["confidence"] = confidence
            state["pred"] = pred.tolist()  # probabilities for ACCEL/BRAKE/NONE
            state["history"] = history


        # 7. sleep a bit to control CPU / match your desired speed
        time.sleep(SLEEP_BETWEEN_FRAMES)

    # cleanup on exit
    keyboard.release(ACCEL_KEY)
    keyboard.release(BRAKE_KEY)
    print("Inference thread stopped.")

# PyQt overlay widget
class OverlayWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        # Make background transparent
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # Let clicks pass through (Qt attribute); may need extra Windows API calls for full click-through
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

        # Small padding inside overlay
        self.padding = 10
        self.font = QtGui.QFont("Consolas", 14, QtGui.QFont.Bold)
        # repaint timer (fast enough)
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.on_update)
        self.update_timer.start(50)  # refresh ~20 fps

        # position update timer (follows game window)
        self.pos_timer = QtCore.QTimer()
        self.pos_timer.timeout.connect(self.update_position_and_size)
        self.pos_timer.start(250)

        # Start with hidden until we find the game window
        self.hide()

    def update_position_and_size(self):
        with state_lock:
            rect = state.get("game_rect")
        if rect is None:
            # try to find game window now
            r = find_game_window()
            if r is None:
                # keep hidden
                self.hide()
                return
            else:
                with state_lock:
                    state["game_rect"] = r
                rect = r

        left, top, width, height = rect
        # set overlay to the same size as the game window
        self.setGeometry(left, top, width, height)
        # show if hidden
        if not self.isVisible():
            self.show()
            try:
                # ensure window is click-through on Windows by setting extended style
                make_window_clickthrough_win(self.winId())
            except Exception:
                # non-fatal; click-through may not be perfect on non-Windows platforms
                pass

    def on_update(self):
        # cause paintEvent
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)

        # --- Panel position ---
        panel_w = 300
        panel_h = 350
        offset_x = 10
        offset_y = 300
        panel_x = self.width() - panel_w - offset_x
        panel_y = self.height() - panel_h - offset_y
        rect = QtCore.QRect(panel_x, panel_y, panel_w, panel_h)

        # semi-transparent black panel
        panel_brush = QtGui.QColor(0, 0, 0, 180)
        painter.fillRect(rect, panel_brush)

        # --- Draw main texts ---
        painter.setFont(self.font)
        text_x = panel_x + 10
        y = panel_y + 28

        with state_lock:
            fps = state.get("fps", 0.0)
            action = state.get("action", "NONE")
            confidence = state.get("confidence", 0.0)
            pred = state.get("pred", [0.0, 0.0, 0.0])
            history = state.get("history", [])

        # Current action (color-coded)
        color_map = {"ACCEL": QtGui.QColor(0, 255, 0), "BRAKE": QtGui.QColor(255, 0, 0), "NONE": QtGui.QColor(255, 255, 0)}
        painter.setPen(color_map.get(action, QtGui.QColor(255, 255, 255)))
        painter.drawText(text_x, y, f"Action: {action}")
        y += 26

        # Confidence of current action
        painter.setPen(QtGui.QColor(180, 220, 255))
        painter.drawText(text_x, y, f"Confidence: {confidence:.2f}")
        y += 26

        # --- Probability bars for all actions ---
        bar_height = 16
        bar_margin = 12
        actions = ["ACCEL", "BRAKE", "NONE"]
        bar_colors = [QtGui.QColor(0, 255, 0), QtGui.QColor(255, 0, 0), QtGui.QColor(255, 255, 0)]

        for i, act in enumerate(actions):
            prob = pred[i] if pred else 0.0

            # Draw action label
            painter.setPen(QtGui.QColor(255, 255, 255))
            painter.drawText(text_x, y, f"{act} {prob:.2f}")

            # Draw bar below label
            bar_y = y + 1
            bar_width = int((panel_w - 40) * prob)
            painter.setBrush(bar_colors[i])
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawRect(text_x, bar_y + 12, bar_width, bar_height)

            # Move y down for next action
            y += bar_height + bar_margin + 20

        # --- Vertical action history ---
        y += 10
        painter.setPen(QtGui.QColor(200, 200, 200))
        painter.drawText(text_x, y, "Prev. Action:")
        y += 20
        max_history = 4
        for past_action in reversed(history[-max_history:]):  # newest on top
            painter.setPen(color_map.get(past_action, QtGui.QColor(255, 255, 255)))
            painter.drawText(text_x + 10, y, past_action)
            y += 20  # vertical spacing between past actions

                # --- Direction indicators (Front/Back) ---
        front_pos = (1130, 670)
        back_pos  = (130, 670)

        # Default colors (white)
        front_color = QtGui.QColor(255, 255, 255)
        back_color = QtGui.QColor(255, 255, 255)

        # Highlight when pressed
        if action == "ACCEL":
            front_color = QtGui.QColor(0, 255, 0)  # green
        elif action == "BRAKE":
            back_color = QtGui.QColor(255, 0, 0)   # red

        arrow_size = 90  # width and height of each square
        corner_radius = 20  # rounded corners

        # Back arrow background
        back_bg_rect = QtCore.QRect(back_pos[0] - 20, back_pos[1] - 70, arrow_size, arrow_size)
        painter.setBrush(QtGui.QColor(0, 0, 0, 150))  # black with transparency
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(back_bg_rect, corner_radius, corner_radius)

        # Front arrow background
        front_bg_rect = QtCore.QRect(front_pos[0] - 20, front_pos[1] - 70, arrow_size, arrow_size)
        painter.setBrush(QtGui.QColor(0, 0, 0, 150))
        painter.drawRoundedRect(front_bg_rect, corner_radius, corner_radius)

        # --- Draw arrows on top ---
        painter.setFont(QtGui.QFont("Consolas", 60, QtGui.QFont.Bold))

        painter.setPen(back_color)
        painter.drawText(back_pos[0], back_pos[1], "<")

        painter.setPen(front_color)
        painter.drawText(front_pos[0], front_pos[1], ">")


        painter.end()


# Windows helper for click-through
def make_window_clickthrough_win(winid):
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    # winId() returns a WId (platform-specific). On Windows this is HWND.
    hwnd = int(winid)

    # Constants
    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    WS_EX_TRANSPARENT = 0x00000020

    # Get current style, add layered + transparent
    old_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    new_style = old_style | WS_EX_LAYERED | WS_EX_TRANSPARENT
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)

    # Ensure layered attributes set (alpha = 255 full opaque for drawing, but background is translucent)
    # SetLayeredWindowAttributes(hwnd, 0, 255, LWA_ALPHA) would be used if you want global alpha.

# Main
def main():
    # Start inference thread
    thr = threading.Thread(target=inference_loop, daemon=True)
    thr.start()

    app = QtWidgets.QApplication(sys.argv)
    overlay = OverlayWidget()

    # ensure the app exits cleanly when interrupted from console (Ctrl+C)
    timer = QtCore.QTimer()
    timer.start(200)
    timer.timeout.connect(lambda: None)

    try:
        app.exec_()
    except KeyboardInterrupt:
        pass
    finally:
        # signal inference to stop
        with state_lock:
            state["running"] = False
        thr.join(timeout=2)
        sys.exit(0)

if __name__ == "__main__":
    main()
