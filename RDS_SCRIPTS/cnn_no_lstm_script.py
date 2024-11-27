# Finder > Go > Connect to server > smb://rds.imperial.ac.uk/rds/user/sa7017/home
# Terminal > ssh sa7017@login.hpc.imperial.ac.uk
#  ~/python3.8/bin/python3 ~/classifier_2_classes/RdsTrainingTemplate.py

import os
# Set the desired directory
desired_path = '/rds/general/user/sa7017/home/BLANKA_DATA/cnn_no_lstm'
# Change the current working directory
os.chdir(desired_path)

import numpy as np
import subprocess
import sys
import importlib

def ensure_package_installed(package_name):
    """
    Ensure the specified package is installed. If not, install it via pip and import it.

    :param package_name: Name of the package to check and install
    :return: The imported package module
    """
    try:
        # Try to import the package
        return importlib.import_module(package_name)
    except ImportError:
        print(f"{package_name} is not installed. Installing now...")
        try:
            # Install the package
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package_name}: {e}")
            sys.exit(1)
        # Try importing the package again after installation
        return importlib.import_module(package_name)

# Ensure TensorFlow is installed and imported
tf = ensure_package_installed("tensorflow")
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Ensure Matplotlib is installed and imported
matplotlib = ensure_package_installed("matplotlib")
import matplotlib.pyplot as plt


import classifier_cnn_no_lstm
from classifier_cnn_no_lstm import build_emg_classifier

folder_path = "/rds/general/user/sa7017/home/BLANKA_DATA/SETS"
for filename in os.listdir(folder_path):
    if filename.endswith(".npy"):  # Only process .npy files
        # Extract variable name from the filename (without extension)
        var_name = os.path.splitext(filename)[0]
        
        # Load the file and handle both dictionary and array cases
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path, allow_pickle=True)  # Load file
        
        # If it's a dictionary, extract using .item()
        if data.shape == ():  # Check if it's a scalar (likely a dictionary)
            globals()[var_name] = data.item()
        else:  # Otherwise, it's a plain NumPy array
            globals()[var_name] = data
        

# Print final shapes of datasets
print("\nFinal Shapes:")
print("X_train: ", X_train.shape, "y_train: ", y_train.shape)
print("X_val: ", X_val.shape, "y_val: ", y_val.shape)
print("X_test: ", X_test.shape, "y_test: ", y_test.shape)


segment_length = 512  # Matches the time dimension after downsampling
num_channels = 16     # Matches the number of PCA-reduced components
num_classes = 3       # Baseline, Preparation, Cancellation

# Build the model
model = build_emg_classifier(segment_length=segment_length, num_channels=num_channels, num_classes=num_classes)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,  # Adjust based on training performance
    batch_size=32,  # Adjust for memory constraints
    verbose=1
)

# Save the model weights
results_path = '/rds/general/user/sa7017/home/BLANKA_DATA/cnn_no_lstm/RESULTS'
model_name = 'classifier_cnn_no_lstm'
model.save_weights(results_path + '/' + model_name + '.weights.h5')
print("\nModel weights saved.")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
# Print the results
print("Test Loss: {:.4f}".format(test_loss))
print("Test Accuracy: {:.2f}".format(test_accuracy * 100))


def plot_training_results(history, save='yes', filename='training_results.png'):
    """
    Visualizes the training and validation accuracy and loss.

    Parameters:
        history: Training history object returned by model.fit().
        save (str): If 'yes', save the plots as an image. Otherwise, display without saving.
        filename (str): Filename for saving the image (default: 'training_results.png').
    """
    # Accuracy plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    if save.lower() == 'yes':
        plt.savefig((filename), dpi=300, bbox_inches='tight')
        print("\nPlot saved.")
        plt.show()
    else:
        plt.show()

plot_training_results(history, save='yes', filename=(results_path + '/' + model_name + '.png'))

# Get predictions for the test set
y_pred = model.predict(X_test)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)



