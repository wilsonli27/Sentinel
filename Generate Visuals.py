import os
import cv2
import numpy as np
import rasterio
import tensorflow as tf
from sklearn.metrics import r2_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress annoying TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
# 🚀 Pointing back to the TRAIN set
TRAIN_DIR = r"D:\Users\Wilson\Downloads\Sentinel\image_cnn\Dataset_Split\train"
MODEL_PATH = "Global_Tillage_DualHead_Final.h5"

BATCH_SIZE = 16
PATCH_SIZE = 64
CLASSES = ['Conservational', 'Conventional', 'Traditional']

# ==========================================
# 2. DATA GENERATOR (Inference Mode)
# ==========================================
class InferencePatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, tiff_dir, batch_size=16, patch_size=64):
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.tiff_files = [os.path.join(tiff_dir, f) for f in os.listdir(tiff_dir) if f.endswith('.tif')]
        print(f"Loaded {len(self.tiff_files)} patches from the TRAIN split.")

    def __len__(self):
        return int(np.ceil(len(self.tiff_files) / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.tiff_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.zeros((len(batch_files), self.patch_size, self.patch_size, 13), dtype=np.float32)
        batch_y_seg = np.zeros((len(batch_files), self.patch_size, self.patch_size, 3), dtype=np.float32)
        batch_y_reg = np.zeros((len(batch_files), 3), dtype=np.float32)
        
        for b, file_path in enumerate(batch_files):
            with rasterio.open(file_path) as src:
                data = src.read() 
                data = np.transpose(data, (1, 2, 0)) 
                data = np.nan_to_num(data, nan=0.0)
                
                h, w, _ = data.shape
                start_y = (h - self.patch_size) // 2
                start_x = (w - self.patch_size) // 2
                data_cropped = data[start_y:start_y+self.patch_size, start_x:start_x+self.patch_size, :]
                
                X_patch = data_cropped[:, :, :13].astype(np.float32)
                Y_patch = data_cropped[:, :, 13:16].astype(np.float32)
                
                X_patch[:, :, :10] /= 10000.0   
                X_patch[:, :, 10:12] /= 30000.0 
                
                ndvi = X_patch[:, :, 7]
                cropland_mask = (ndvi > -0.1) & (ndvi < 0.8)
                cropland_mask = np.expand_dims(cropland_mask, axis=-1)
                X_patch = X_patch * cropland_mask
                
                batch_x[b] = X_patch
                batch_y_seg[b] = Y_patch
                batch_y_reg[b] = np.mean(Y_patch, axis=(0, 1)) 
                
        return batch_x, {'seg_output': batch_y_seg, 'reg_output': batch_y_reg}

# ==========================================
# 3. EXTRACTION FUNCTION
# ==========================================
def extract_predictions(model, data_gen):
    print("\nExtracting Predictions from Dual-Head Model...")
    y_true_reg = {0: [], 1: [], 2: []}
    y_pred_reg = {0: [], 1: [], 2: []}
    
    for i in range(len(data_gen)):
        X_batch, Y_dict = data_gen[i]
        true_fractions = Y_dict['reg_output']
        
        preds = model.predict(X_batch, verbose=0)
        reg_preds = preds[1] 
        
        for c in range(3):
            y_true_reg[c].extend(true_fractions[:, c])
            y_pred_reg[c].extend(reg_preds[:, c])
            
    return y_true_reg, y_pred_reg

# ==========================================
# 6. SPATIAL GRAD-CAM (3-IMAGE MAXIMUM CONCENTRATION SCANNER)
# ==========================================
def generate_gradcam(model, data_gen):
    print("\n[3/3] Generating Spatial Grad-CAM (Scanning for the 3 best individual patches...)")
    
    best_imgs = []
    best_ys = []
    
    # --- PHASE 1: Scan for the maximum concentration of each class ---
    for c in range(3):
        best_img, best_y, highest_frac = None, None, -1.0
        print(f"  -> Hunting for high {CLASSES[c]} presence...")
        
        # Scan through the first 50 batches to find the most concentrated patch
        for i in range(min(50, len(data_gen))): 
            X_batch, Y_dict = data_gen[i]
            Y_batch_seg = Y_dict['seg_output']
            
            for b in range(X_batch.shape[0]):
                # Make sure the patch isn't just black/empty space
                if np.count_nonzero(X_batch[b, :, :, 0]) > (PATCH_SIZE * PATCH_SIZE * 0.3):
                    frac = np.mean(Y_batch_seg[b, :, :, c])
                    if frac > highest_frac:
                        highest_frac = frac
                        best_img = X_batch[b:b+1]
                        best_y = Y_batch_seg[b]
                        
        best_imgs.append(best_img)
        best_ys.append(best_y)
        print(f"     ✓ Found perfect patch! (True {CLASSES[c]} Fraction: {highest_frac:.2f})")

    # --- PHASE 2: Generate class-specific Grad-CAMs ---
    seg_layer = model.get_layer('seg_output')
    last_conv_features = seg_layer.input
    
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[last_conv_features, seg_layer.output]
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Spatial Grad-CAM: Class-Specific Targeting", fontsize=16)

    for c in range(3):
        img = best_imgs[c]
        y_true_seg = best_ys[c]

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            # 🚀 We ask the AI to calculate gradients specifically for class 'c'
            loss = predictions[:, :, :, c] 

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (64, 64))
        
        r_band = img[0, :, :, 2]
        g_band = img[0, :, :, 1]
        b_band = img[0, :, :, 0]
        rgb_img = np.dstack((r_band, g_band, b_band))
        rgb_img = np.clip(rgb_img * 3.0, 0, 1) 
        
        # Top Row: The actual RGB satellite image
        axes[0, c].imshow(rgb_img)
        axes[0, c].set_title(f"Targeting: {CLASSES[c]}\n(True Fraction in this image: {np.mean(y_true_seg[:, :, c]):.2f})")
        axes[0, c].axis('off')
        
        # Bottom Row: The AI's heat map overlay
        axes[1, c].imshow(rgb_img, alpha=0.5)
        im = axes[1, c].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[1, c].set_title(f"AI Focus Map\n(Where is the {CLASSES[c]}?)")
        axes[1, c].axis('off')
        
    plt.tight_layout()
    plt.savefig("FINAL_Train_Dual_GradCAM_3Images.png", dpi=300)
    print("✓ Saved FINAL_Train_Dual_GradCAM_3Images.png")
    plt.show()
# ==========================================
# 7. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Loading Dual-Head ResNet-50 Model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False) 
    
    train_gen = InferencePatchGenerator(TRAIN_DIR, BATCH_SIZE, PATCH_SIZE)
    
    y_true_reg, y_pred_reg = extract_predictions(model, train_gen)

    generate_gradcam(model, train_gen)
    
    print("\n🎉 Visual suite complete!")