"""
ECG CNN Training Pipeline

End-to-end training script that:
1. Loads all patient records from data/raw/
2. Preprocesses & segments heartbeats
3. Trains the 1D-CNN with early stopping
4. Saves the best model to models/
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocess import ECGPreprocessor
from src.model import build_1d_cnn


class Annotation:
    """Simple container for ECG beat annotations."""
    def __init__(self, samples, symbols):
        self.sample = np.array(samples)
        self.symbol = symbols


def load_annotation(path):
    """Parse annotation text file from MIT-BIH database."""
    ann_data = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    ann_data.append((int(parts[1]), parts[2]))
                except ValueError:
                    continue
    return Annotation([x[0] for x in ann_data], [x[1] for x in ann_data])


def load_all_records(data_dir, preprocessor):
    """
    Load and preprocess all patient records.
    
    Args:
        data_dir: Path to raw data directory
        preprocessor: ECGPreprocessor instance
    
    Returns:
        X: Feature array (n_samples, 180)
        y: Label array (n_samples,)
    """
    all_X, all_y = [], []
    
    csv_files = sorted([f.replace('.csv', '') for f in os.listdir(data_dir) if f.endswith('.csv')])
    print(f"📂 Found {len(csv_files)} patient records")
    
    for record_id in csv_files:
        try:
            # Load signal
            df = pd.read_csv(os.path.join(data_dir, f'{record_id}.csv'))
            df.columns = [col.strip("'") for col in df.columns]
            signal = df['MLII'].values
            
            # Load annotations
            ann_path = os.path.join(data_dir, f'{record_id}annotations.txt')
            annotation = load_annotation(ann_path)
            
            # Preprocess
            filtered = preprocessor.apply_filter(signal)
            beats, peaks = preprocessor.segment_beats(filtered)
            X, y = preprocessor.map_labels(beats, peaks, annotation)
            
            all_X.append(X)
            all_y.append(y)
            print(f"  ✅ Record {record_id}: {len(X)} beats")
            
        except Exception as e:
            print(f"  ❌ Record {record_id}: {e}")
    
    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    
    return X, y


def train_model(X, y, epochs=50, batch_size=64, test_size=0.2, save_dir='models'):
    """
    Train the 1D-CNN model.
    
    Args:
        X: Feature array
        y: Label array
        epochs: Number of training epochs
        batch_size: Batch size
        test_size: Fraction for test split
        save_dir: Directory to save model and artifacts
    
    Returns:
        model: Trained Keras model
        history: Training history
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Reshape for CNN: (samples, timesteps, channels)
    X_reshaped = X.reshape(-1, 180, 1)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training:   {X_train.shape[0]} samples")
    print(f"   Testing:    {X_test.shape[0]} samples")
    
    # Class weights for imbalanced medical data
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {int(i): float(w) for i, w in enumerate(weights)}
    print(f"   ⚖️ Class Weights: {class_weights}")
    
    # Build model
    num_classes = len(np.unique(y))
    model = build_1d_cnn(input_shape=(180, 1), num_classes=num_classes)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(save_dir, 'best_ecg_model.h5'),
            monitor='val_accuracy', save_best_only=True
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]
    
    # Train
    print(f"\n🚀 Training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n🎯 Final Test Accuracy: {accuracy:.2%}")
    print(f"   Final Test Loss: {loss:.4f}")
    print(f"   Model saved to: {save_dir}/best_ecg_model.h5")
    
    return model, history, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    # Configuration
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    # Initialize preprocessor
    preprocessor = ECGPreprocessor(fs=360, window_size=180)
    
    # Load all data
    X, y = load_all_records(DATA_DIR, preprocessor)
    
    print(f"\n📈 Total Dataset: {X.shape[0]} heartbeats")
    classes, counts = np.unique(y, return_counts=True)
    class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Paced']
    for c, n in zip(classes, counts):
        print(f"   {class_names[c]}: {n} ({n/len(y)*100:.1f}%)")
    
    # Train
    model, history, splits = train_model(X, y, epochs=50, save_dir=SAVE_DIR)
    
    print("\n✅ Training pipeline complete!")
