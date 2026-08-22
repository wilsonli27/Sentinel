import os
import cv2
import numpy as np
import rasterio
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Input, MaxPooling2D, UpSampling2D, Concatenate, GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import ResNet50
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
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

BATCH_SIZE = 16
PATCH_SIZE = 64
STEPS_PER_EPOCH = 150 
EPOCHS = 75 

CLASSES = ['Conservational', 'Conventional', 'Traditional']

print("=" * 60)
print("INITIALIZING DUAL-HEAD RESNET-50 WITH FOCAL TVERSKY LOSS")
print("=" * 60)

# ==========================================
# 2. CUSTOM LOSS: FOCAL TVERSKY
# ==========================================
def focal_tversky_loss(alpha=0.3, beta=0.7, gamma=1.33):
    # Alpha = 0.3 (Lower penalty for missing a pixel)
    # Beta = 0.7 (Massive penalty for hallucinating/bleeding outside boundaries)
    def loss(y_true, y_pred):
        smooth = 1e-6
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        
        # LABEL SMOOTHING: Prevents the Softmax from rounding to hyper-confident 1.0/0.0
        epsilon = 0.1
        y_true_smooth = y_true * (1.0 - epsilon) + (epsilon / 3.0)
        
        y_true_idx = tf.argmax(y_true_smooth, axis=-1)
        y_true_hard = tf.one_hot(y_true_idx, depth=3)
        
        tversky_total = 0.0
        # 3.0 weight ensures Conservation is STILL targeted, but Beta 0.7 keeps it from bleeding
        class_weights = [3.0, 1.0, 1.0] 
        
        for c in range(3):
            y_t = K.flatten(y_true_hard[:, :, :, c])
            y_p = K.flatten(y_pred[:, :, :, c])
            
            TP = K.sum(y_t * y_p)
            FP = K.sum((1 - y_t) * y_p)
            FN = K.sum(y_t * (1 - y_p))
            
            tversky = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)
            tversky_total += class_weights[c] * K.pow((1.0 - tversky), gamma)
            
        return tversky_total / K.sum(tf.constant(class_weights, dtype=tf.float32))
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
# 3. DUAL-HEAD DATA GENERATOR
# ==========================================
class DualHeadPatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, tiff_dir, batch_size=16, patch_size=64, steps_per_epoch=100, is_training=False, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.steps_per_epoch = steps_per_epoch
        self.is_training = is_training
        self.tiff_files = [os.path.join(tiff_dir, f) for f in os.listdir(tiff_dir) if f.endswith('.tif')]

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        batch_x = np.zeros((self.batch_size, self.patch_size, self.patch_size, 13), dtype=np.float32)
        batch_y_seg = np.zeros((self.batch_size, self.patch_size, self.patch_size, 3), dtype=np.float32)
        batch_y_reg = np.zeros((self.batch_size, 3), dtype=np.float32) # New target for the R2 Head
        
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
                    if np.random.rand() > 0.5: X_patch = np.flip(X_patch, axis=1); Y_patch = np.flip(Y_patch, axis=1)
                    if np.random.rand() > 0.5: X_patch = np.flip(X_patch, axis=0); Y_patch = np.flip(Y_patch, axis=0)
                    k_rot = np.random.randint(0, 4)
                    X_patch = np.rot90(X_patch, k_rot, axes=(0, 1)); Y_patch = np.rot90(Y_patch, k_rot, axes=(0, 1))
                
                ndvi = X_patch[:, :, 7]
                cropland_mask = (ndvi > -0.1) & (ndvi < 0.8)
                cropland_mask = np.expand_dims(cropland_mask, axis=-1)
                X_patch = X_patch * cropland_mask
                
                batch_x[b] = X_patch
                batch_y_seg[b] = Y_patch
                batch_y_reg[b] = np.mean(Y_patch, axis=(0, 1)) # Collapse the 64x64 into 3 exact fractional averages
                b += 1
                
        # Return a dictionary targeting both neural heads
        return batch_x, {'seg_output': batch_y_seg, 'reg_output': batch_y_reg}

# ==========================================
# 4. DUAL-HEAD RESNET-50 ARCHITECTURE
# ==========================================
def build_dual_head_resnet50(input_shape=(PATCH_SIZE, PATCH_SIZE, 13)):
    inputs = Input(shape=input_shape)
    
    adapter = Conv2D(3, (1, 1), padding='same', use_bias=False, name='channel_adapter')(inputs)
    resnet_base = ResNet50(include_top=False, weights='imagenet', input_shape=(PATCH_SIZE, PATCH_SIZE, 3))
    
    skip_layer_names = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    resnet_outputs = [resnet_base.get_layer(name).output for name in skip_layer_names]
    resnet_extractor = Model(inputs=resnet_base.input, outputs=resnet_outputs, name="resnet_extractor")
    
    s1, s2, s3, s4, b1 = resnet_extractor(adapter)
    
    # --- HEAD 1: SEGMENTATION (The Map Drawer) ---
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
    c10 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(c10)
    c10 = BatchNormalization()(c10)
    
    outputs_seg = Conv2D(3, (1, 1), activation='softmax', name='seg_output')(c10)
    
    # --- HEAD 2: REGRESSION (The Fraction Calculator) ---
    # Tap directly into the ResNet bottleneck (b1) before it gets upsampled
    flat = GlobalAveragePooling2D()(b1)
    d1 = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(flat)
    d1 = Dropout(0.3)(d1)
    outputs_reg = Dense(3, activation='softmax', name='reg_output')(d1)
    
    # Build Model with 2 Outputs
    model = Model(inputs=inputs, outputs=[outputs_seg, outputs_reg])
    
    # Compile with two completely separate loss functions
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
        loss={
            'seg_output': focal_tversky_loss(alpha=0.3, beta=0.7), # Strict Map Drawing
            'reg_output': 'mae'                                    # Strict Fraction R2 Math
        },
        loss_weights={'seg_output': 1.0, 'reg_output': 0.5}, # Balance the gradients
        metrics={
            'seg_output': [CustomMeanIoU(), tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
            'reg_output': [tf.keras.metrics.MeanAbsoluteError(name='mae')]
        }
    )
    return model

# ==========================================
# 5. VISUALIZATIONS (Dual-Head Output)
# ==========================================
def plot_dual_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # Plot 1: Segmentation Tversky Loss
    axes[0].plot(history.history['seg_output_loss'], label='Seg Train Loss', color='blue')
    axes[0].plot(history.history['val_seg_output_loss'], label='Seg Val Loss', color='orange')
    axes[0].set_title('Focal Tversky Loss (Shapes)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Regression MAE Loss
    axes[1].plot(history.history['reg_output_mae'], label='Reg Train MAE', color='purple')
    axes[1].plot(history.history['val_reg_output_mae'], label='Reg Val MAE', color='magenta')
    axes[1].set_title('Fractional Error (R-Squared Driver)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Mean IoU
    axes[2].plot(history.history['seg_output_mean_iou'], label='Train Mean IoU', color='green')
    axes[2].plot(history.history['val_seg_output_mean_iou'], label='Val Mean IoU', color='red')
    axes[2].set_title('Mean IoU')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("Training_History_DualHead.png", dpi=300)
    print("✓ Saved Training_History_DualHead.png")
    plt.close()

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    train_gen = DualHeadPatchGenerator(TRAIN_DIR, BATCH_SIZE, PATCH_SIZE, STEPS_PER_EPOCH, is_training=True)
    val_gen = DualHeadPatchGenerator(VAL_DIR, BATCH_SIZE, PATCH_SIZE, steps_per_epoch=50, is_training=False)
    
    model = build_dual_head_resnet50()
    
    # The monitor variable must target the combined primary metric
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
    print(f"\n🚀 Commencing Dual-Head Training Run ({EPOCHS} Epochs)...")
    history = model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=EPOCHS, 
        callbacks=[lr_scheduler, early_stop]
    )
    
    model.save("Global_Tillage_DualHead_Final.h5")
    print("\n✅ Training Complete. Model saved as 'Global_Tillage_DualHead_Final.h5'.")
    
    plot_dual_history(history)