import tensorflow as tf
from tensorflow.keras.layers import ( # type: ignore
    Conv1D, MaxPooling1D, BatchNormalization, LSTM, Bidirectional,
    Dropout, Dense, Input, GlobalAveragePooling1D, Attention, Add, UpSampling1D
)
from tensorflow.keras.models import Model # type: ignore

# Input Shape Parameters
segment_length = 512  # Matches the time dimension after downsampling
num_channels = 16     # Matches the number of PCA-reduced components
num_classes = 2       # Preparation, Cancellation

def build_emg_classifier(segment_length, num_channels, num_classes):
    """
    CNN-Bidirectional LSTM classifier with attention for EMG signal classification.

    Enhanced with:
    • Expanded CNN kernel sizes and added convolutional layer for better feature extraction.
    • Residual connections between CNN layers to preserve critical features.
    • Attention mechanism to focus on critical time steps.
    """
    # Input layer
    inputs = Input(shape=(segment_length, num_channels), name="InputLayer")
    
    # CNN Block: Feature extraction
    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same', name="Conv1D_1")(inputs)
    x = BatchNormalization(name="BatchNorm_1")(x)
    x = MaxPooling1D(pool_size=2, strides=2, name="MaxPooling1D_1")(x)
    x = Dropout(0.2, name="Dropout_1")(x)

    # Residual CNN Block
    cnn_residual = Conv1D(filters=128, kernel_size=7, activation='relu', padding='same', name="Conv1D_2")(x)
    cnn_residual = BatchNormalization(name="BatchNorm_2")(cnn_residual)
    cnn_residual = MaxPooling1D(pool_size=2, strides=2, name="MaxPooling1D_2")(cnn_residual)

    # Align sequence length
    cnn_residual = UpSampling1D(size=2, name="Residual_UpSampling1D")(cnn_residual)  # Upsample to match sequence length

    # Align channel dimensions
    cnn_residual = Conv1D(filters=64, kernel_size=1, activation=None, padding='same', name="Residual_Conv1D")(cnn_residual)  # Match channels

    # Add residual connection
    x = Add(name="CNN_Residual_Add")([x, cnn_residual])
    x = Dropout(0.3, name="Dropout_2")(x)

    # Additional Convolutional Layer
    x = Conv1D(filters=256, kernel_size=9, activation='relu', padding='same', name="Conv1D_3")(x)
    x = BatchNormalization(name="BatchNorm_3")(x)
    x = MaxPooling1D(pool_size=2, strides=2, name="MaxPooling1D_3")(x)
    x = Dropout(0.3, name="Dropout_3")(x)

    # Bidirectional LSTM Block: Temporal dependencies
    lstm_out = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2), name="BiLSTM_1")(x)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2), name="BiLSTM_2")(lstm_out)

    # Attention Mechanism: Focus on important time steps
    attention_scores = tf.keras.layers.Attention(name="Attention")([lstm_out, lstm_out])
    attention_output = Add(name="AddAttention")([lstm_out, attention_scores])

    # Global Pooling
    pooled = GlobalAveragePooling1D(name="GlobalAveragePooling")(attention_output)

    # Fully Connected Classifier
    x = Dense(128, activation='relu', name="Dense_1")(pooled)
    x = Dropout(0.4, name="Dropout_4")(x)
    x = Dense(64, activation='relu', name="Dense_2")(x)
    x = Dropout(0.4, name="Dropout_5")(x)
    
    # Output layer for binary classification
    outputs = Dense(2, activation='softmax', name="OutputLayer")(x)

    # Build the model
    model = Model(inputs, outputs, name="EMG_Classifier_Attention")

    return model

# Instantiate the model
model = build_emg_classifier(segment_length, num_channels, num_classes)

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # If one-hot encoded labels
    metrics=['accuracy']
)

# Print the model summary
model.summary()