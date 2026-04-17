"""
ECG CNN Evaluation Module

Provides functions for:
- Confusion matrix visualization
- Classification report (precision, recall, F1)
- Training history plots
- Saliency map visualization (CNN attention)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report


# AAMI beat class names
CLASS_NAMES = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Paced']


def evaluate_model(model, X_test, y_test):
    """
    Run full evaluation on the test set.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        y_pred: Predicted labels
        report: Classification report string
    """
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    used_names = [CLASS_NAMES[i] for i in unique_labels]
    
    report = classification_report(
        y_test, y_pred, 
        labels=unique_labels, 
        target_names=used_names
    )
    
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    
    print("=" * 60)
    print("       ARRHYTHMIA DETECTION — EVALUATION RESULTS")
    print("=" * 60)
    print(report)
    print(f"🎯 Overall Test Accuracy: {accuracy:.2%}")
    
    return y_pred, report


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    """
    Plot and optionally save the confusion matrix.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        save_path: Optional path to save the figure
    """
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    used_names = [CLASS_NAMES[i] for i in unique_labels]
    
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=used_names, yticklabels=used_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('🫀 ECG Arrhythmia Detection — Confusion Matrix', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training accuracy and loss curves.
    
    Args:
        history: Keras training history object
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📈 Training history saved to: {save_path}")
    
    plt.show()


def saliency_map(model, input_signal, class_idx):
    """
    Compute saliency map — gradient of output w.r.t. input.
    
    Shows which parts of the ECG signal are most important
    for the model's prediction.
    
    Args:
        model: Trained Keras model
        input_signal: Single sample (1, 180, 1)
        class_idx: Target class index
    
    Returns:
        saliency: Normalized importance array
    """
    input_tensor = tf.Variable(input_signal, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor, training=False)
        target = predictions[:, class_idx]
    
    grads = tape.gradient(target, input_tensor)
    saliency = tf.abs(grads).numpy().flatten()
    saliency = saliency / (saliency.max() + 1e-8)
    
    return saliency


def plot_saliency_maps(model, X_test, y_test, y_pred, save_path=None):
    """
    Plot saliency maps for each beat class.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: True labels
        y_pred: Predicted labels
        save_path: Optional path to save the figure
    """
    unique_classes = np.unique(y_test)
    fig, axes = plt.subplots(len(unique_classes), 1, 
                              figsize=(14, 3.5 * len(unique_classes)))
    
    for idx, cls in enumerate(unique_classes):
        mask = (y_test == cls) & (y_pred == cls)
        if np.sum(mask) == 0:
            axes[idx].set_title(f'{CLASS_NAMES[cls]} — No correct predictions')
            continue
        
        sample_idx = np.where(mask)[0][0]
        sample = X_test[sample_idx:sample_idx + 1]
        signal = sample.flatten()
        
        # Compute saliency
        importance = saliency_map(model, sample, cls)
        
        ax = axes[idx]
        time = np.arange(len(signal))
        ax.plot(time, signal, color='black', linewidth=1.2, label='ECG Signal')
        
        # Highlight important regions
        for j in range(len(signal) - 1):
            if importance[j] > 0.3:
                ax.axvspan(j, j + 1, alpha=importance[j] * 0.5, color='red')
        
        high_imp = importance > 0.5
        ax.scatter(time[high_imp], signal[high_imp], c='red', s=10, zorder=5, alpha=0.6)
        
        ax.set_title(f'🫀 {CLASS_NAMES[cls]} Beat — CNN Attention',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(0, len(signal))
    
    axes[-1].set_xlabel('Sample Points (360 Hz)')
    plt.suptitle('Where Does the CNN Look to Detect Arrhythmias?',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"🧠 Saliency maps saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    import os
    
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_ecg_model.h5')
    SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    print("ECG CNN Evaluation Module")
    print("=" * 40)
    print("Available functions:")
    print("  - evaluate_model(model, X_test, y_test)")
    print("  - plot_confusion_matrix(y_test, y_pred)")
    print("  - plot_training_history(history)")
    print("  - plot_saliency_maps(model, X_test, y_test, y_pred)")
