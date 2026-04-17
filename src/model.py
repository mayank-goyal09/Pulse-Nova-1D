import tensorflow as tf
from tensorflow.keras import layers, models

def build_1d_cnn(input_shape, num_classes):
    model = models.Sequential([
        # Block 1
        layers.Conv1D(32, kernel_size=5, padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=2),
        
        # Block 2
        layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Block 3
        layers.Conv1D(128, kernel_size=5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(), # Smarter than 'Flatten'
        
        # Output
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Initialize
input_shape = (180, 1) # 180 samples, 1 channel (MLII)
model = build_1d_cnn(input_shape, num_classes=5)
model.summary()