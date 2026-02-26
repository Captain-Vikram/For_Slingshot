<p align="center">
  <img src="demo/demo.gif" alt="Demo" width="600" />
</p>

# 🧠 Hill Climb Racing AI — Vision-Based Driving Agent

This project implements a **supervised deep learning system** that learns to autonomously control a vehicle in the *Countryside* level of **Hill Climb Racing**.  
Using computer vision and a convolutional neural network (CNN), the model observes raw gameplay frames and predicts the correct driving action in real time — mimicking human input.

The model learns directly from **labeled gameplay footage**, where each frame is tagged with one of three possible player actions:
- `Accelerate` (0)
- `Brake` (1)
- `None` (2)

It effectively performs **behavioral cloning** — learning to replicate the driving behavior of a human player based solely on visual input.


## 🚗 Overview

The pipeline for this project consists of these key stages, from data collection to deployment:

### 1. 🎮 Data Capture  
Gameplay frames are recorded from the Countryside level using a custom frame-grabbing script. Each frame is automatically paired with the corresponding **keyboard input** (accelerate, brake, or no action).  
To prevent bias and irrelevant learning, all non-gameplay elements such as **fuel bars, coins, score counters, and pedals** are **blacked out**.

### 2. 🧹 Preprocessing  
Captured frames are resized to **64×64 RGB images** to reduce computation while retaining enough detail for decision-making. The dataset is then saved as a compressed `.npz` archive for efficient loading.

```python
data = np.load("hcr_dataset.npz")
X, y = data["X"], data["y"]
```

### 3. 🧠 Model Training
A Convolutional Neural Network (CNN) was designed and trained using TensorFlow/Keras.
The architecture balances simplicity and efficiency, using only 7 layers — sufficient for pattern recognition without overfitting.

| Layer | Type         | Details                                 |
| :---: | :----------- | :-------------------------------------- |
|   1   | Conv2D       | 16 filters, 3×3 kernel, ReLU            |
|   2   | MaxPooling2D | 2×2 pool                                |
|   3   | Conv2D       | 32 filters, 3×3 kernel, ReLU            |
|   4   | MaxPooling2D | 2×2 pool                                |
|   5   | Flatten      | —                                       |
|   6   | Dense        | 64 neurons, ReLU                        |
|   7   | Dense        | 3 neurons, Softmax (ACCEL, BRAKE, NONE) |

### 3. 🕹️ Real-Time Inference

Once trained, the model can be connected to a screen-capture overlay system that:
1. Continuously reads the game window
2. Resizes each frame to the CNN input size
3. Predicts the next driving action
4. Simulates key presses (Accelerate or Brake)
5. Displays an overlay showing predicted action, FPS, and confidence
6. This allows the AI to autonomously play the game, reacting to slopes, fuel tanks, and obstacles.

## 📂 Project Structure

```
├── model/
│ ├── best_model.h5 
│ └── final_model.h5
├── scripts/
│ ├── datacapture.py 
│ ├── cleanandfilter.py
│ ├── loadandsave.py 
│ ├── splitandtrain.py 
│ └── run.py 
└── README.md
```

## 📊 Dataset

**Dataset name:** Hill Climb Racing AI Dataset - Countryside

This dataset contains **29,278 gameplay frames** from the Countryside level of *Hill Climb Racing*.

Each frame is labeled with one of three possible driving actions:

| Label | Action     |
| :---: | :---------- |
| **0** | Accelerate  |
| **1** | Brake       |
| **2** | None        |

📦 **File:** `hcr_dataset.npz`
- `X` → `(29278, 64, 64, 3)` — RGB gameplay frames  
- `y` → `(29278,)` — integer labels (0–2)

All UI elements such as **fuel bar, pedals, score, coins, and timers** are blacked out to avoid distracting features and ensure consistent learning.

👉 **I have uploaded the dataset here:** https://www.kaggle.com/datasets/fahzainsaiyed/hill-climb-racing-gameplay-countryside


## 🧩 Model

**Model name:** Hill Climb Racing AI Model (Countryside)

A small and efficient CNN trained from scratch to predict driving actions based on screen input.

| Layer | Type | Details |
|:------:|:-----|:--------|
| 1 | Conv2D | 16 filters, 3×3 kernel, ReLU |
| 2 | MaxPooling2D | 2×2 pool |
| 3 | Conv2D | 32 filters, 3×3 kernel, ReLU |
| 4 | MaxPooling2D | 2×2 pool |
| 5 | Flatten | — |
| 6 | Dense | 64 neurons, ReLU |
| 7 | Dense | 3 neurons, Softmax (ACCEL, BRAKE, NONE) |


👉 **I have uploaded the model here:** https://www.kaggle.com/models/fahzainsaiyed/hillclimber


### 🚀 Usage

You have **two ways** to use this project — either **run the pre-trained model immediately** or **train your own model from scratch** using the provided tools.


### 🧩 Option 1 — Run Pre-Trained Model (Quick Start)

If you just want to see the AI in action:

1. Open **Hill Climb Racing**.
2. Run the following script:

   ```bash
   python scripts/run.py

The model (best_model.h5) will automatically load and start controlling the game in real-time.
- The AI observes the screen.
- It predicts the driving action (Accelerate, Brake, or None).
- It presses keys to play autonomously.
- On-screen overlay displays FPS, latency, and prediction info.

✅ No extra setup needed. Everything required to run the model is already included.

### ⚙️ Option 2 — Train Your Own Model

If you want to retrain the model using your own gameplay, follow these steps:

1️⃣ Capture Data
Run the data capture script while playing Hill Climb Racing:
```bash
python scripts/datacapture.py
```

This records gameplay frames and logs your key inputs.
All raw images will be saved into a hcr_data/ folder.

2️⃣ Clean & Mask Frames
After collecting frames, preprocess them to remove HUD elements (fuel bar, coins, score, pedals, etc.) by blacking them out:
```bash
python scripts/cleanerfilter.py
```
This produces masked gameplay frames in hcr_data_masked/.

3️⃣ Build the Dataset
Convert the preprocessed images into a compressed NumPy dataset:
```bash
python scripts/loadandsave.py
```

This will create a single file:
hcr_dataset.npz

containing:
X: preprocessed images (shape (N, 64, 64, 3))
y: driving action labels (0, 1, 2)

4️⃣ Train the Model
Train the CNN on your dataset:
```bash
python scripts/splitandtrain.py
```

This script will:
- Split data into training and validation sets (80/20)
- Compute class weights to balance uneven action data
- Train the CNN with early stopping and model checkpointing

Save:
best_model.h5 — best validation accuracy
final_model.h5 — last trained model

5️⃣ Run Your Trained Model

Finally, launch the agent with your custom-trained model:
```bash
python scripts/run.py
```

The script automatically detects your saved model and uses it for inference.


| Step | Script             | Description                   |
| ---- | ------------------ | ----------------------------- |
| 1    | `datacapture.py`   | Capture frames and key inputs |
| 2    | `cleanerfilter.py` | Mask out HUD elements         |
| 3    | `loadandsave.py`   | Build and save `.npz` dataset |
| 4    | `splitandtrain.py` | Train CNN and save best model |
| 5    | `run.py`           | Run AI agent in real-time     |

