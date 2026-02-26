import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# LOAD DATASET
data = np.load("./data/hcr_dataset.npz")
X = data['X']
y = data['y']

print(f"Dataset loaded: {X.shape[0]} samples")

# SPLIT DATASET
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]} | Validation samples: {X_val.shape[0]}")

# COMPUTE CLASS WEIGHTS
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes, class_weights))
print("Class weights:", class_weights)
# Example output: {0: 0.8, 1: 1.5, 2: 0.9}

# BUILD CNN
IMG_SIZE = X.shape[1]  # The images are square 64x64

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # ACCEL, BRAKE, NONE
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# CALLBACKS: SAVE BEST + EARLY STOP
checkpoint = ModelCheckpoint(
    "./model/best_model.h5",       # file name for best model
    monitor="val_accuracy",        # metric to monitor
    save_best_only=True,           # only save when val_accuracy improves
    mode="max",                    # maximize accuracy
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_accuracy",        # stop when validation accuracy stops improving
    patience=3,                    # number of epochs to wait before stopping
    mode="max",
    restore_best_weights=True,     # load best weights after stopping
    verbose=1
)

# TRAIN MODEL (WITH CLASS WEIGHTS)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,         # early stopping will stop automatically
    batch_size=16,
    callbacks=[checkpoint, early_stop],
    class_weight=class_weights     # <--- THIS IS THE KEY ADDITION
)

# SAVE FINAL MODEL
model.save("./model/final_model.h5")
print("Training complete. Best model saved as best_model.h5")
