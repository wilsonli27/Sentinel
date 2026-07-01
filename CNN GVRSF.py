import os
import cv2
import numpy as np
import rasterio
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Input, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import ResNet50
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress annoying TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
BASE_DIR = r"D:\Users\Wilson\Downloads\Sentinel\image_cnn\Dataset_Split"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

BATCH_SIZE = 16
PATCH_SIZE = 64
STEPS_PER_EPOCH = 150 
EPOCHS = 75 

CLASSES = ['Conservational', 'Conventional', 'Traditional']

print(f"TensorFlow Version: {tf.__version__}")
print("=" * 60)
print("INITIALIZING RESNET-50 U-NET WITH BINARIZED COMBO LOSS")
print("=" * 60)

# ==========================================
# 2. CUSTOM COMBO LOSS (BINARIZED DICE + SOFT MAE)
# ==========================================
def combo_dice_mae_loss(class_weights=[5.0, 1.0, 1.0]): 
    # Weight relaxed to 5.0 for stability to prevent bleed
    weights = tf.constant(class_weights, dtype=tf.float32)
    
    def loss(y_true, y_pred):
        smooth = 1e-6
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        
        # --- 1. BINARIZE LABELS FOR DICE (Shape tracing) ---
        # Temporarily convert [0.1, 0.8, 0.1] into [0.0, 1.0, 0.0] so the Dice math can reach 1.0
        y_true_idx = tf.argmax(y_true, axis=-1)
        y_true_hard = tf.one_hot(y_true_idx, depth=3)
        
        dice_loss_total = 0.0
        for c in range(3):
            # Calculate Dice strictly on the Hard Labels vs AI Predictions
            y_true_c = K.flatten(y_true_hard[:, :, :, c])
            y_pred_c = K.flatten(y_pred[:, :, :, c])
            
            intersection = K.sum(y_true_c * y_pred_c)
            dice = (2. * intersection + smooth) / (K.sum(y_true_c) + K.sum(y_pred_c) + smooth)
            dice_loss_total += weights[c] * (1.0 - dice)
            
        dice_loss = dice_loss_total / K.sum(weights)
        
        # --- 2. SOFT MEAN ABSOLUTE ERROR (Fractional R^2 Optimization) ---
        # Calculate MAE using the original fractional soft labels to push the R^2 up
        mae_loss = K.mean(K.abs(y_true - y_pred))
        
        # Combine: 70% Shape (Dice), 30% Fractions (MAE)
        return dice_loss + (0.3 * mae_loss)
        
    return loss

class CustomMeanIoU(tf.keras.metrics.Metric):
    def __init__(self, name='mean_iou', **kwargs):
        super(CustomMeanIoU, self).__init__(name=name, **kwargs)
        self.iou_metric = tf.keras.metrics.MeanIoU(num_classes=3)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_idx = tf.argmax(y_true, axis=-1)
        y_pred_idx = tf.argmax(y_pred, axis=-1)
        self.iou_metric.update_state(y_true_idx, y_pred_idx)

    def result(self):
        return self.iou_metric.result()

    def reset_state(self):
        self.iou_metric.reset_state()

# ==========================================
# 3. DYNAMIC DATA GENERATOR
# ==========================================
class DynamicPatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, tiff_dir, batch_size=16, patch_size=64, steps_per_epoch=100, is_training=False, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.steps_per_epoch = steps_per_epoch
        self.is_training = is_training
        self.tiff_files = [os.path.join(tiff_dir, f) for f in os.listdir(tiff_dir) if f.endswith('.tif')]
        print(f"Generator initialized: {len(self.tiff_files)} patches. (Augmentation: {self.is_training})")

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        batch_x = np.zeros((self.batch_size, self.patch_size, self.patch_size, 13), dtype=np.float32)
        batch_y = np.zeros((self.batch_size, self.patch_size, self.patch_size, 3), dtype=np.float32)
        
        b = 0
        while b < self.batch_size:
            file_path = np.random.choice(self.tiff_files)
            
            with rasterio.open(file_path) as src:
                data = src.read() 
                data = np.transpose(data, (1, 2, 0)) 
                data = np.nan_to_num(data, nan=0.0)
                
                h, w, _ = data.shape
                if h < self.patch_size or w < self.patch_size: continue 
                
                start_y = (h - self.patch_size) // 2
                start_x = (w - self.patch_size) // 2
                data_cropped = data[start_y:start_y+self.patch_size, start_x:start_x+self.patch_size, :]
                
                X_patch = data_cropped[:, :, :13].astype(np.float32)
                Y_patch = data_cropped[:, :, 13:16].astype(np.float32)
                
                if np.all(X_patch[:, :, :10] == 0.0): continue 
                
                X_patch[:, :, :10] /= 10000.0   
                X_patch[:, :, 10:12] /= 30000.0 
                
                if self.is_training:
                    if np.random.rand() > 0.5:
                        X_patch = np.flip(X_patch, axis=1)
                        Y_patch = np.flip(Y_patch, axis=1)
                    if np.random.rand() > 0.5:
                        X_patch = np.flip(X_patch, axis=0)
                        Y_patch = np.flip(Y_patch, axis=0)
                    k_rot = np.random.randint(0, 4)
                    X_patch = np.rot90(X_patch, k_rot, axes=(0, 1))
                    Y_patch = np.rot90(Y_patch, k_rot, axes=(0, 1))
                
                ndvi = X_patch[:, :, 7]
                cropland_mask = (ndvi > -0.1) & (ndvi < 0.8)
                cropland_mask = np.expand_dims(cropland_mask, axis=-1)
                X_patch = X_patch * cropland_mask
                
                batch_x[b] = X_patch
                batch_y[b] = Y_patch
                b += 1
                
        return batch_x, batch_y

# ==========================================
# 4. PRE-TRAINED RESNET-50 BACKBONE ARCHITECTURE
# ==========================================
def build_resnet50_unet(input_shape=(PATCH_SIZE, PATCH_SIZE, 13)):
    inputs = Input(shape=input_shape)
    
    adapter = Conv2D(3, (1, 1), padding='same', use_bias=False, name='channel_adapter')(inputs)
    resnet_base = ResNet50(include_top=False, weights='imagenet', input_shape=(PATCH_SIZE, PATCH_SIZE, 3))
    
    skip_layer_names = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    resnet_outputs = [resnet_base.get_layer(name).output for name in skip_layer_names]
    resnet_extractor = Model(inputs=resnet_base.input, outputs=resnet_outputs, name="resnet_extractor")
    
    s1, s2, s3, s4, b1 = resnet_extractor(adapter)
    
    u1 = UpSampling2D((2, 2))(b1) 
    c6 = Concatenate()([u1, s4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(c6)
    c6 = BatchNormalization()(c6)
    
    u2 = UpSampling2D((2, 2))(c6) 
    c7 = Concatenate()([u2, s3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(c7)
    c7 = BatchNormalization()(c7)
    
    u3 = UpSampling2D((2, 2))(c7) 
    c8 = Concatenate()([u3, s2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(c8)
    c8 = BatchNormalization()(c8)
    
    u4 = UpSampling2D((2, 2))(c8) 
    c9 = Concatenate()([u4, s1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(c9)
    c9 = BatchNormalization()(c9)
    
    u5 = UpSampling2D((2, 2))(c9) 
    c10 = Concatenate()([u5, adapter])
    c10 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4), name='last_conv')(c10)
    c10 = BatchNormalization()(c10)
    
    outputs = Conv2D(3, (1, 1), activation='softmax')(c10)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
        loss=combo_dice_mae_loss(class_weights=[5.0, 1.0, 1.0]), 
        metrics=[CustomMeanIoU(), tf.keras.metrics.CategoricalAccuracy(name='accuracy')] 
    )
    return model

# ==========================================
# 5. VISUALIZATIONS (Tri-Metric Output)
# ==========================================
def plot_training_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # Plot 1: Loss
    axes[0].plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    axes[0].set_title('Combo Binarized Dice + Soft MAE Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', color='purple', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='magenta', linewidth=2)
    axes[1].set_title('Pixel Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Mean IoU
    axes[2].plot(history.history['mean_iou'], label='Training IoU', color='green', linewidth=2)
    axes[2].plot(history.history['val_mean_iou'], label='Validation IoU', color='red', linewidth=2)
    axes[2].set_title('Mean IoU (Shape Overlap)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Mean IoU')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("Training_History_ResNet_Dice.png", dpi=300)
    print("✓ Saved Training_History_ResNet_Dice.png")
    plt.close()

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    train_gen = DynamicPatchGenerator(TRAIN_DIR, BATCH_SIZE, PATCH_SIZE, STEPS_PER_EPOCH, is_training=True)
    val_gen = DynamicPatchGenerator(VAL_DIR, BATCH_SIZE, PATCH_SIZE, steps_per_epoch=50, is_training=False)
    
    model = build_resnet50_unet()
    
    # 🚀 Increased patience to 5 to give the optimizer more runway before slashing the LR
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
    print(f"\n🚀 Commencing ResNet-50 Training with Binarized Combo Loss ({EPOCHS} Epochs)...")
    history = model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=EPOCHS, 
        callbacks=[lr_scheduler, early_stop]
    )
    
    model.save("Global_Tillage_ResNet50_Dice.h5")
    print("\n✅ Training Complete. Model saved as 'Global_Tillage_ResNet50_Dice.h5'.")
    
    plot_training_history(history)