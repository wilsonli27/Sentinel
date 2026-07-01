import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, GlobalAveragePooling2D
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, GlobalAveragePooling2D
import numpy as np
import os
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CUSTOM DATA LOADER (For .npy files)
# ==========================================
def load_spectral_data(data_dir, classes):
    images = []
    labels = []
    
    label_map = {name: i for i, name in enumerate(classes)}
    
    for c in classes:
        path = os.path.join(data_dir, c)
        files = [f for f in os.listdir(path) if f.endswith('.npy')]
        print(f"Loading {len(files)} images for {c}...")
        
        for f in files:
            img_array = np.load(os.path.join(path, f))
            
            # SCALING:
            # Your data is Int16 (0-10000). CNN needs 0-1.
            img_array = img_array.astype('float32') / 10000.0
            
            images.append(img_array)
            labels.append(label_map[c])
            
    return np.array(images), np.array(labels)

CLASSES = ["Conventional", "Conservational", "Rotational"]
DATA_DIR = "data"

X, y = load_spectral_data(DATA_DIR, CLASSES)

# Split Data (80/20 as per paper guidelines [cite: 87])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Shape: {X_train.shape}") # Should be (N, 64, 64, 10)

# ==========================================
# 2. MODEL ARCHITECTURE (Referencing PDF)
# ==========================================
# Input shape: 64x64 with 10 bands (B2-B12 + Indices)
input_shape = X_train.shape[1:] 

model = Sequential()

# Block 1 [cite: 94, 96]
model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
model.add(BatchNormalization()) # [cite: 97]
model.add(MaxPooling2D((2,2)))  # [cite: 103]
model.add(Dropout(0.25))        # [cite: 108]

# Block 2
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

# Block 3
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

# Block 4 [cite: 104]
model.add(Conv2D(256, (3,3), activation='relu', padding='same', name="last_conv"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

# Classifier Head
# Paper uses Global Average Pooling instead of Flatten [cite: 105]
model.add(GlobalAveragePooling2D()) 

model.add(Dense(128, activation='relu')) # [cite: 106]
model.add(BatchNormalization())
model.add(Dropout(0.5)) # [cite: 106]

# Output Layer (3 Classes)
model.add(Dense(3, activation='softmax'))

# Compile
# Adam optimizer with lr=0.001 [cite: 109]
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==========================================
# 3. TRAINING
# ==========================================
# Paper uses callbacks for ReduceLROnPlateau and EarlyStopping [cite: 125]
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
]

history = model.fit(
    X_train, y_train,
    epochs=40,               # [cite: 127]
    batch_size=32,           # [cite: 127]
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# ==========================================
# 4. PLOTTING
# ==========================================
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.show()