import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

class ECGPreprocessor:
    def __init__(self, fs=360, window_size=180):
        self.fs = fs
        self.window_size = window_size
        self.half_window = window_size // 2

    def apply_filter(self, signal):
        """Removes baseline wander and high-frequency noise."""
        nyq = 0.5 * self.fs
        low = 0.5 / nyq
        high = 40.0 / nyq
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def segment_beats(self, signal):
        """Locates R-peaks and slices the signal into windows."""
        # Height threshold: 20% of the max signal to avoid noise peaks
        peaks, _ = find_peaks(signal, distance=200, height=0.2 * np.max(signal))
        
        beats = []
        valid_peak_indices = []
        
        for p in peaks:
            if p > self.half_window and p < len(signal) - self.half_window:
                segment = signal[p - self.half_window : p + self.half_window]
                # Z-score Normalization: Professional standard for ML input
                segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
                beats.append(segment)
                valid_peak_indices.append(p)
                
        return np.array(beats), np.array(valid_peak_indices)

    def map_labels(self, beats, peak_indices, annotation):
        """Groups complex medical codes into 5 AAMI categories."""
        label_map = {
            'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
            'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular
            'V': 2, 'E': 2,                          # Ventricular
            'F': 3,                                  # Fusion
            '/': 4, 'f': 4, 'Q': 4                   # Paced/Unknown
        }
        
        X, y = [], []
        ann_indices = annotation.sample
        ann_symbols = annotation.symbol
        
        for i, peak in enumerate(peak_indices):
            if i >= len(beats):
                break
            # Find the closest annotation within a 20-sample tolerance
            closest_idx = np.argmin(np.abs(ann_indices - peak))
            if np.abs(ann_indices[closest_idx] - peak) < 20:
                symbol = ann_symbols[closest_idx]
                if symbol in label_map:
                    X.append(beats[i])
                    y.append(label_map[symbol])
                    
        return np.array(X), np.array(y)