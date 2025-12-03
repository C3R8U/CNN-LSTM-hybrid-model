# CNN-LSTM-hybrid-model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Global Configuration & Reproducibility
# ==========================================
def set_global_seed(seed=42):
    """
    Sets random seeds for reproducibility across numpy, python, and tensorflow.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[INFO] Global random seed set to: {seed}")

PARAMS = {
    'sampling_rate': 1000,      # Hz (Original)
    'seq_length': 300,          # Downsampled length for model input (Time steps)
    'n_channels': 2,            # ECG and GSR
    'n_classes': 4,             # No, Mild, Moderate, Severe
    'batch_size': 32,
    'epochs': 50,               # Reduced for demonstration (Paper uses more)
    'learning_rate': 0.002,
    'n_subjects': 160,
    'n_splits': 5               # 5-Fold CV
}

# ==========================================
# 2. Data Simulation (Placeholder for Real Data)
# ==========================================
def generate_synthetic_data(n_subjects=160, seq_len=300):
    """
    Generates synthetic dummy data to mimic ECG and GSR signals.
    Returns:
        X: (N_samples, seq_len, 2)
        y: (N_samples,)
        groups: (N_samples,) -> Subject IDs for cross-validation
    """
    print("[INFO] Generating synthetic dataset...")
    X_list = []
    y_list = []
    groups_list = []
    
    samples_per_subject = 30 # Simulate 30 segments per child
    
    for subject_id in range(n_subjects):
        # Assign a random anxiety level to the subject (0-3)
        label = np.random.randint(0, 4)
        
        for _ in range(samples_per_subject):
            # 1. Simulate ECG (Sine wave with noise + localized peaks)
            t = np.linspace(0, 10, seq_len)
            ecg = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 2.5 * t)
            # Add "QRS-like" spikes
            spike_indices = np.random.choice(seq_len, size=5, replace=False)
            ecg[spike_indices] += 3.0
            ecg += np.random.normal(0, 0.2, seq_len) # Noise
            
            # 2. Simulate GSR (Slow drift + peaks)
            gsr = np.cumsum(np.random.normal(0, 0.1, seq_len)) # Random walk
            # Add stress peaks based on label (higher label = more peaks)
            if label > 1:
                stress_indices = np.random.choice(seq_len, size=label*3, replace=False)
                gsr[stress_indices] += 2.0
            
            # Combine
            signal = np.stack([ecg, gsr], axis=1) # Shape (300, 2)
            
            X_list.append(signal)
            y_list.append(label)
            groups_list.append(subject_id)
            
    X = np.array(X_list)
    y = np.array(y_list)
    groups = np.array(groups_list)
    
    print(f"[INFO] Dataset shape: {X.shape}, Labels shape: {y.shape}")
    return X, y, groups

# ==========================================
# 3. Signal Preprocessing
# ==========================================
def butter_filter(data, cutoff, fs, order=3, btype='low'):
    """
    Applies Butterworth filter.
    """
    nyq = 0.5 * fs
    if isinstance(cutoff, list):
        normal_cutoff = [c / nyq for c in cutoff]
    else:
        normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y

def preprocess_signals(X):
    """
    Applies filtering and normalization.
    """
    print("[INFO] Preprocessing signals (Filtering & Normalization)...")
    X_processed = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        # Channel 0: ECG (Bandpass 0.5-45Hz)
        X_processed[i, :, 0] = butter_filter(X[i, :, 0], [0.5, 45], 
                                             fs=100, # Assuming downsampled fs for this example
                                             btype='band')
        
        # Channel 1: GSR (Lowpass 0.2Hz)
        X_processed[i, :, 1] = butter_filter(X[i, :, 1], 0.2, 
                                             fs=100, 
                                             btype='low')
        
        # Z-score Normalization
        for c in range(2):
            mean = np.mean(X_processed[i, :, c])
            std = np.std(X_processed[i, :, c]) + 1e-6
            X_processed[i, :, c] = (X_processed[i, :, c] - mean) / std
            
    return X_processed

# ==========================================
# 4. CNN-LSTM Model Architecture
# ==========================================
def build_cnn_lstm_model(input_shape, n_classes):
    """
    Constructs the hybrid CNN-LSTM model as described in the paper.
    """
    inputs = layers.Input(shape=input_shape)
    
    # --- Feature Extraction (CNN) ---
    # Conv Layer 1
    x = layers.Conv1D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    
    # Conv Layer 2
    x = layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    
    # Conv Layer 3
    x = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    
    # --- Temporal Analysis (LSTM) ---
    # No Flatten() here. We keep the time dimension for LSTM.
    # Shape entering LSTM: (Batch, Reduced_Steps, 64)
    x = layers.LSTM(units=128, return_sequences=False, dropout=0.5)(x)
    
    # --- Classification ---
    x = layers.Dense(128, activation='elu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_Hybrid")
    
    optimizer = optimizers.Adam(learning_rate=PARAMS['learning_rate'])
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# ==========================================
# 5. Explainability (Grad-CAM 1D)
# ==========================================
class GradCAM1D:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = models.Model(
            [model.inputs], 
            [model.get_layer(layer_name).output, model.output]
        )

    def compute_heatmap(self, input_data, class_idx):
        with tf.GradientTape() as tape:
            inputs = tf.cast(input_data, tf.float32)
            conv_outputs, predictions = self.grad_model(inputs)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

# ==========================================
# 6. Main Evaluation Pipeline (5-Fold CV)
# ==========================================
def main():
    set_global_seed(42)
    
    # 1. Get Data
    X, y, groups = generate_synthetic_data(PARAMS['n_subjects'], PARAMS['seq_length'])
    X = preprocess_signals(X)
    y_cat = to_categorical(y, PARAMS['n_classes'])
    
    # 2. Setup Cross-Validation (GroupKFold ensures no subject overlap)
    gkf = GroupKFold(n_splits=PARAMS['n_splits'])
    
    fold_metrics = []
    confusion_matrices = []
    
    print("\n[INFO] Starting 5-Fold Cross-Validation (Subject-wise)...")
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n--- Fold {fold+1}/{PARAMS['n_splits']} ---")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_cat[train_idx], y_cat[test_idx]
        
        # Build Model
        model = build_cnn_lstm_model((PARAMS['seq_length'], PARAMS['n_channels']), PARAMS['n_classes'])
        
        # Callbacks (Early Stopping)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        # Train
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=PARAMS['epochs'],
            batch_size=PARAMS['batch_size'],
            callbacks=[early_stopping],
            verbose=0 # Silent training for cleaner output
        )
        
        # Evaluate
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        
        try:
            auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
        except ValueError:
            auc = 0.0 # Handle edge cases
            
        print(f"Fold {fold+1} Result -> Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        fold_metrics.append([acc, prec, rec, f1, auc])
        confusion_matrices.append(confusion_matrix(y_true, y_pred))

        # --- Grad-CAM Demo for the last fold ---
        if fold == 4:
            print("[INFO] Generating Grad-CAM for a sample in Fold 5...")
            # Pick a "Severe Anxiety" sample (Class 3)
            sample_idx = np.where(y_true == 3)[0][0]
            input_sample = X_test[sample_idx:sample_idx+1]
            
            # Use the last convolutional layer name (usually 'conv1d_2' based on architecture)
            # Note: In real scenarios, check model.summary() for exact layer name
            grad_cam = GradCAM1D(model, layer_name='conv1d_2') 
            heatmap = grad_cam.compute_heatmap(input_sample, class_idx=3)
            print(f"Grad-CAM heatmap generated (Shape: {heatmap.shape})")

    # 3. Aggregated Results
    metrics_np = np.array(fold_metrics)
    mean_metrics = np.mean(metrics_np, axis=0)
    std_metrics = np.std(metrics_np, axis=0)
    
    print("\n" + "="*50)
    print("FINAL RESULTS (Mean +/- SD over 5 Folds)")
    print("="*50)
    print(f"Accuracy:  {mean_metrics[0]*100:.2f}% (+/- {std_metrics[0]*100:.2f}%)")
    print(f"Precision: {mean_metrics[1]*100:.2f}% (+/- {std_metrics[1]*100:.2f}%)")
    print(f"Recall:    {mean_metrics[2]*100:.2f}% (+/- {std_metrics[2]*100:.2f}%)")
    print(f"F1-Score:  {mean_metrics[3]*100:.2f}% (+/- {std_metrics[3]*100:.2f}%)")
    print(f"AUROC:     {mean_metrics[4]:.4f}     (+/- {std_metrics[4]:.4f})")
    print("="*50)
    
    # Summing Confusion Matrices
    total_cm = np.sum(confusion_matrices, axis=0)
    print("\nAggregated Confusion Matrix:")
    print(total_cm)

if __name__ == "__main__":
    main()
