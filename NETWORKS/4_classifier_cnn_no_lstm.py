import tensorflow as tf
from tensorflow.keras.layers import (  # type: ignore
    Conv1D, MaxPooling1D, BatchNormalization, Dropout, Dense, Input,
    GlobalAveragePooling1D, Add, UpSampling1D
)
from tensorflow.keras.models import Model # type: ignore

# Input Shape Parameters
segment_length = 512  # Matches the time dimension after downsampling
num_channels = 16     # Matches the number of PCA-reduced components
num_classes = 3       # Baseline, Preparation, Cancellation

def build_emg_classifier(segment_length, num_channels, num_classes):
    """
    Simplified CNN-based EMG classifier for three-class classification.
    
    Key Features:
    • Convolutional layers to extract temporal features from EMG signals.
    • Residual connections to preserve critical features across layers.
    • Increased dropout rates to reduce overfitting.
    • Dense layers for classification.
    """
    # Input layer
    inputs = Input(shape=(segment_length, num_channels), name="InputLayer")
    
    # CNN Block: Initial Feature Extraction
    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same', name="Conv1D_1")(inputs)
    x = BatchNormalization(name="BatchNorm_1")(x)
    x = MaxPooling1D(pool_size=2, strides=2, name="MaxPooling1D_1")(x)
    x = Dropout(0.3, name="Dropout_1")(x)

    # Residual CNN Block
    cnn_residual = Conv1D(filters=128, kernel_size=7, activation='relu', padding='same', name="Conv1D_2")(x)
    cnn_residual = BatchNormalization(name="BatchNorm_2")(cnn_residual)
    cnn_residual = MaxPooling1D(pool_size=2, strides=2, name="MaxPooling1D_2")(cnn_residual)

    # Align sequence length and channel dimensions for residual connection
    cnn_residual = UpSampling1D(size=2, name="Residual_UpSampling1D")(cnn_residual)  # Align temporal dimension
    cnn_residual = Conv1D(filters=64, kernel_size=1, activation=None, padding='same', name="Residual_Conv1D")(cnn_residual)  # Align channels

    # Add residual connection
    x = Add(name="CNN_Residual_Add")([x, cnn_residual])
    x = Dropout(0.4, name="Dropout_2")(x)

    # Additional Convolutional Layer
    x = Conv1D(filters=256, kernel_size=9, activation='relu', padding='same', name="Conv1D_3")(x)
    x = BatchNormalization(name="BatchNorm_3")(x)
    x = MaxPooling1D(pool_size=2, strides=2, name="MaxPooling1D_3")(x)
    x = Dropout(0.5, name="Dropout_3")(x)

    # Global Pooling
    pooled = GlobalAveragePooling1D(name="GlobalAveragePooling")(x)

    # Fully Connected Classifier
    x = Dense(128, activation='relu', name="Dense_1")(pooled)
    x = Dropout(0.5, name="Dropout_4")(x)
    x = Dense(64, activation='relu', name="Dense_2")(x)
    x = Dropout(0.5, name="Dropout_5")(x)
    
    # Output layer with softmax for classification
    outputs = Dense(num_classes, activation='softmax', name="OutputLayer")(x)

    # Build the model
    model = Model(inputs, outputs, name="EMG_Classifier_Simple")

    return model

# Build the model
model = build_emg_classifier(segment_length=segment_length, num_channels=num_channels, num_classes=num_classes)

# # Compile the model
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # Print the model summary
# model.summary()