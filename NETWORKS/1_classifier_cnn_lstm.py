# Key Additions and Updates

# 	1.	Downsampled Input:
# 	•	Reason: Preprocessed input downsampled to 512 Hz, retaining beta-band activity while reducing sequence length and computational cost.
# 	2.	Larger CNN Kernel Sizes:
# 	•	Larger kernel sizes (e.g., 5, 7, 9) align with beta-band oscillations (~33–77 ms periods), improving the model’s ability to detect relevant frequency components.
# 	3.	Attention Mechanism:
# 	•	Added an attention mechanism to focus on time steps with the most discriminative information (e.g., transitions between classes).
# 	•	Implemented with Attention and Add layers to weight and refine LSTM outputs.
# 	4.	Global Average Pooling:
# 	•	Averages across time steps in the LSTM output, condensing the sequence into a fixed-size feature vector for the dense layers.
# 	5.	Fully Connected Layers:
# 	•	Two dense layers with dropout for robust high-level representation learning before classification.

import tensorflow as tf
from tensorflow.keras.layers import ( # type: ignore
    Conv1D, MaxPooling1D, BatchNormalization, LSTM, Bidirectional,
    Dropout, Dense, Input, GlobalAveragePooling1D, Attention, Add, Flatten
)
from tensorflow.keras.models import Model # type: ignore

# Input Shape Parameters
segment_length = 512  # Matches the time dimension after downsampling
num_channels = 16     # Matches the number of PCA-reduced components
num_classes = 3       # Baseline, Preparation, Cancellation

# Define the architecture
def build_emg_classifier(segment_length, num_channels, num_classes):
    """
    CNN-Bidirectional LSTM classifier with attention for EMG signal classification.

    • CNNs for extracting local temporal features (e.g., beta-band oscillations),
    • Bidirectional LSTMs for capturing long-term dependencies (e.g., transitions between states),
    • Attention mechanism to focus on critical time steps. 
    • Global Average Pooling to reduce LSTM outputs to a fixed-size vector
    • Dense layers classify the integrated spatial-temporal features.
    """
    # Input layer
    inputs = Input(shape=(segment_length, num_channels), name="InputLayer")
    
    # CNN Block: Feature extraction
    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same', name="Conv1D_1")(inputs)
    x = BatchNormalization(name="BatchNorm_1")(x)
    x = MaxPooling1D(pool_size=2, strides=2, name="MaxPooling1D_1")(x)
    x = Dropout(0.2, name="Dropout_1")(x)
    
    x = Conv1D(filters=128, kernel_size=7, activation='relu', padding='same', name="Conv1D_2")(x)
    x = BatchNormalization(name="BatchNorm_2")(x)
    x = MaxPooling1D(pool_size=2, strides=2, name="MaxPooling1D_2")(x)
    x = Dropout(0.3, name="Dropout_2")(x)
    
    x = Conv1D(filters=256, kernel_size=9, activation='relu', padding='same', name="Conv1D_3")(x)
    x = BatchNormalization(name="BatchNorm_3")(x)
    x = MaxPooling1D(pool_size=2, strides=2, name="MaxPooling1D_3")(x)
    x = Dropout(0.3, name="Dropout_3")(x)

    # Bidirectional LSTM Block: Temporal dependencies
    lstm_out = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), name="BiLSTM_1")(x)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), name="BiLSTM_2")(lstm_out)

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
    
    # Output layer with softmax for classification
    outputs = Dense(num_classes, activation='softmax', name="OutputLayer")(x)

    # Build the model
    model = Model(inputs, outputs, name="EMG_Classifier_Attention")

    return model
