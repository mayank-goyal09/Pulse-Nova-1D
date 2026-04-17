"""
ECG CNN - Electrocardiogram Classification using 1D Convolutional Neural Networks

This package provides tools for:
- Signal preprocessing (filtering, R-peak detection, segmentation)
- 1D-CNN model architecture for heartbeat classification
- Training pipeline with early stopping and checkpointing
- Evaluation metrics and saliency map visualization
"""

from .preprocess import ECGPreprocessor
from .model import build_1d_cnn
from .evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_training_history,
    plot_saliency_maps
)

__version__ = "1.0.0"
__all__ = [
    "ECGPreprocessor",
    "build_1d_cnn",
    "evaluate_model",
    "plot_confusion_matrix",
    "plot_training_history",
    "plot_saliency_maps",
]
