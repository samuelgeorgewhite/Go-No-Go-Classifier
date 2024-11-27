"""
# 1. Load data (EMG signals, states, triggers)

# 2. Segment trials into Baseline, Preparation (for No-Go trials), and Cancellation windows

# 3. Reduce channels to 16 using PCA

# 4. Apple bandpass filter (13–30 Hz) to focus on beta-band and downsample (new_sr = 512 Hz)

# 5. Normalize (MVC and z-score)
emg_normalized = normalize_to_mvc(emg_downsampled)
emg_standardized = standardize_across_trials(emg_normalized)

# 6. Data augmentation (jitter, slight augmentations to frequency filtering)

# 8. Split into train, validation, test sets (by participant)

# 9. Prepare processed data for classifier training

"""

import h5py
import json
import os


def load_mat_files(file_paths):
    """
    Load one, or multiple .mat files from a directory, and store their contents in a dictionary to avoid overwriting keys.
    
    Parameters:
        file_paths (str or list): Path(s) to .mat files (can be a single file or a list of files).
    
    Returns:
        dict: A dictionary containing the data from each .mat file.
    """
    # Ensure file_paths is a list (in cases where single paths are given)
    if isinstance(file_paths, str) and os.path.isdir(file_paths):
        file_paths = [os.path.join(file_paths, f) for f in os.listdir(file_paths) if f.endswith('.mat')]
    elif isinstance(file_paths, str):  # If a single file path is given
        file_paths = [file_paths]

    all_data = {}  # Dictionary to store all .mat file contents

    for file_path in file_paths:
        file_name = os.path.basename(file_path)  # Use the file name as the key
        subject_code = file_name.split('_')[0]
        all_data[subject_code] = {}  # Nested dictionary for this file

        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                if f[key].shape == ():  # Scalar dataset (e.g., JSON string)
                    value = f[key][()]  # Access scalar dataset
                    if key == 'states':  # If it's the `states` key, deserialize the JSON
                        value = json.loads(value.decode('utf-8'))  # Decode and load JSON
                    all_data[subject_code][key] = value
                else:  # Non-scalar dataset
                    all_data[subject_code][key] = f[key][:]
    
    return all_data

# # Directory of .mat files
# file_paths = '/Users/samuelking/Library/CloudStorage/OneDrive-ImperialCollegeLondon/UNI/PhD/CODE/BLANKA_DATA/DATA/RAW'
# all_data = load_mat_files(file_paths)


# Segment DATA INTO BASELINE, PREPARATION AND CANCELLATION CLASSES

def segment_data(all_data, baseline_window=(-2, -1), preparation_window=(-1, 0), cancellation_window=(0, 1)):
    """
    Segment the EMG data into Baseline, Preparation, and Cancellation classes based on trigger locations,
    with Cancellation only applicable for NO-GO trials (state_vec == 1).

    Parameters:
        all_data (dict): Dictionary containing EMG data and metadata for all participants.
        baseline_window (tuple): Time window (in seconds) for the Baseline class relative to triggers.
        preparation_window (tuple): Time window (in seconds) for the Preparation class relative to triggers.
        cancellation_window (tuple): Time window (in seconds) for the Cancellation class relative to triggers.

    Returns:
        dict: A dictionary with segmented data for each class and participant.
    """
    segmented_data = {}

    for subject_code, data in all_data.items():
        # Extract relevant variables
        emg = data['EMG']  # Shape: (num_samples, num_channels)
        trg_loc = data['trg_loc'].flatten()  # Trigger locations (sample indices)
        state_vec = data['state_vec'].flatten()  # State at every timestep
        fs = int(data['fsamp'].flatten()[0])  # Sampling frequency (e.g., 2048 Hz)

        # Convert time windows to sample counts
        baseline_samples = (int(baseline_window[0] * fs), int(baseline_window[1] * fs))
        preparation_samples = (int(preparation_window[0] * fs), int(preparation_window[1] * fs))
        cancellation_samples = (int(cancellation_window[0] * fs), int(cancellation_window[1] * fs))

        # Initialize segmented data structure for this participant
        segmented_data[subject_code] = {
            'Baseline': [],
            'Preparation': [],
            'Cancellation': []
        }

        # Iterate through each trigger location
        for trigger in trg_loc:
            # Baseline window
            baseline_start = int(trigger + baseline_samples[0])
            baseline_end = int(trigger + baseline_samples[1])
            if 0 <= baseline_start < emg.shape[0] and 0 <= baseline_end < emg.shape[0]:
                segmented_data[subject_code]['Baseline'].append(emg[baseline_start:baseline_end, :])

            # Preparation window
            prep_start = int(trigger + preparation_samples[0])
            prep_end = int(trigger + preparation_samples[1])
            if 0 <= prep_start < emg.shape[0] and 0 <= prep_end < emg.shape[0]:
                segmented_data[subject_code]['Preparation'].append(emg[prep_start:prep_end, :])

            # Cancellation window (only for NO-GO trials where state_vec == 1)
            if state_vec[int(trigger)] == 1:  # Ensure the state at the trigger is NO-GO
                cancel_start = int(trigger + cancellation_samples[0])
                cancel_end = int(trigger + cancellation_samples[1])
                if 0 <= cancel_start < emg.shape[0] and 0 <= cancel_end < emg.shape[0]:
                    segmented_data[subject_code]['Cancellation'].append(emg[cancel_start:cancel_end, :])

    return segmented_data

# segmented_data = segment_data(all_data, baseline_window=(-2, -1), preparation_window=(-1, 0), cancellation_window=(0, 1))


def reshape_emg_data(trial, num_channels=64):
    """
    Reshape EMG data from 256 channels into smaller groups of `num_channels` channels.

    Parameters:
        trial (np.array): A single trial of EMG data with shape (timesteps, 256).
        num_channels (int): Number of channels per group (default: 64).

    Returns:
        np.array: Reshaped EMG data with shape (timesteps, num_groups, group_size).
    """

    num_groups = trial.shape[1] // num_channels

    # Ensure the number of channels is divisible by group_size
    assert trial.shape[1] % num_channels == 0, "Number of channels must be divisible by group_size"

    # Reshape the trial into groups
    reshaped_trial = trial.reshape(-1, num_groups, num_channels)  # Shape: (timesteps, num_groups, group_size)
    return reshaped_trial


from sklearn.decomposition import PCA
import numpy as np

def apply_pca_to_groups(reshaped_trial, n_components=16):
    """
    Apply PCA to each group of EMG channels and reduce the dimensionality.

    Parameters:
        reshaped_trial (np.array): Reshaped EMG data with shape (timesteps, num_groups, num_channels).
        n_components (int): Number of principal components to retain per group.

    Returns:
        np.array: PCA-reduced EMG data with shape (timesteps, num_groups * n_components).
    """
    num_groups = reshaped_trial.shape[1]
    reduced_groups = []

    for group in range(num_groups):
        # Extract a single group (timesteps, group_size)
        group_data = reshaped_trial[:, group, :]

        # Apply PCA to the group
        pca = PCA(n_components=n_components)
        reduced_group = pca.fit_transform(group_data)  # Shape: (timesteps, n_components)

        # # Log variance retained for debugging
        # explained_variance = sum(pca.explained_variance_ratio_)
        # print(f"Group {group + 1} PCA retained variance: {explained_variance:.2f}")

        reduced_groups.append(reduced_group)

    # Concatenate all reduced groups into a single array
    reduced_trial = np.stack(reduced_groups, axis=1)

    return reduced_trial # Shape: (timesteps, num_groups, n_components)

def reshape_and_reduce_channels(segmented_data, num_chanels=64, n_components=16):
    """
    Reshape EMG data and apply PCA to reduce dimensionality.

    Parameters:
        segmented_data (dict): Segmented EMG data dictionary.
        num_chanels (int): Number of initial channels per group (default: 64).
        n_components (int): Number of principal components to retain per group.

    Returns:
        dict: EMG data with reduced channels for each segment.
    """
    reduced_data = {}

    for subject_code, segments in segmented_data.items():
        reduced_data[subject_code] = {}

        for segment_type, trials in segments.items():  # Iterate over Baseline, Preparation, Cancellation
            reduced_data[subject_code][segment_type] = []

            for trial in trials:
                # Reshape the trial into groups of `group_size` channels
                reshaped_trial = reshape_emg_data(trial, num_channels=num_chanels)

                # Apply PCA to reduce each group to `n_components` principal components
                reduced_trial = apply_pca_to_groups(reshaped_trial, n_components=n_components)

                # Store the reduced trial
                reduced_data[subject_code][segment_type].append(reduced_trial)

    return reduced_data # shape (samples, no. of groups, no. of PC's (key channels))

# reduced_data = reshape_and_reduce_channels(segmented_data, num_chanels=64, n_components=16)



# 4. Bandpass filter (13–30 Hz) and downsample (512 Hz)

from scipy.signal import butter, filtfilt

def bandpass_filter(data, low_cutoff, high_cutoff, fs):
    """
    Apply a Butterworth bandpass filter to the data.

    Parameters:
        data (np.array): The input data array with shape (samples, no. of groups, no. of PCs).
        low_cutoff (float): Lower cutoff frequency (in Hz).
        high_cutoff (float): Upper cutoff frequency (in Hz).
        fs (int): Sampling frequency of the data (in Hz).

    Returns:
        np.array: Bandpass-filtered data with the same shape as input.
    """
    # Design the Butterworth filter
    b, a = butter(N=4, Wn=[low_cutoff, high_cutoff], btype='band', fs=fs)
    
    # Apply the filter to each group and PC
    filtered_data = np.zeros_like(data)
    for group in range(data.shape[1]):
        for pc in range(data.shape[2]):
            filtered_data[:, group, pc] = filtfilt(b, a, data[:, group, pc])
    
    return filtered_data


def downsample_data(data, original_fs, target_fs):
    """
    Downsample the data to a lower sampling rate. (In this case, the 13–30 Hz bandpass filter prior
    already isolates the beta-band activity, which is well below the Nyquist frequency of the target 
    sampling rate (512 Hz). This makes this simple downsampling method appropriate.)

    Parameters:
        data (np.array): The input data array with shape (samples, no. of groups, no. of PCs).
        original_fs (int): Original sampling frequency of the data (in Hz).
        target_fs (int): Target sampling frequency (in Hz).

    Returns:
        np.array: Downsampled data.
    """
    downsample_factor = original_fs // target_fs
    if original_fs % target_fs != 0:
        raise ValueError("The original sampling rate must be an integer multiple of the target sampling rate.")
    
    # Downsample along the samples dimension
    downsampled_data = data[::downsample_factor, :, :]
    return downsampled_data


def filter_and_downsample_data(reduced_data, fs, target_fs=512, low_cutoff=13, high_cutoff=30):
    """
    Preprocess all participants and classes in the reduced_data dictionary by applying
    bandpass filtering and downsampling.

    Parameters:
        reduced_data (dict): Dictionary with reduced data for each participant and class.
                             Shape of arrays: (samples, num_groups, num_components).
        fs (int): Original sampling frequency of the data (in Hz).
        target_fs (int): Target sampling frequency (default: 512 Hz).
        low_cutoff (float): Lower cutoff frequency for bandpass filter (default: 13 Hz).
        high_cutoff (float): Upper cutoff frequency for bandpass filter (default: 30 Hz).

    Returns:
        dict: Preprocessed data with the same structure as the input but filtered and downsampled.
    """
    preprocessed_data = {}

    for subject_code, classes in reduced_data.items():
        preprocessed_data[subject_code] = {}

        for class_label, trials in classes.items():
            preprocessed_data[subject_code][class_label] = []

            for trial in trials:
                # Apply bandpass filter
                filtered_trial = bandpass_filter(trial, low_cutoff, high_cutoff, fs)

                # Downsample the trial
                downsampled_trial = downsample_data(filtered_trial, original_fs=fs, target_fs=target_fs)

                # Append the preprocessed trial
                preprocessed_data[subject_code][class_label].append(downsampled_trial)
                filtered_and_downsampled_data = preprocessed_data

    return filtered_and_downsampled_data

"""
# 5. Normalize (MVC and z-score)
emg_normalized = normalize_to_mvc(emg_downsampled)
emg_standardized = standardize_across_trials(emg_normalized)
"""

def normalize_to_mvc(data, mvc_value=None):
    """
    Normalize EMG data to a percentage of the Maximum Voluntary Contraction (MVC).

    Parameters:
        data (np.array): EMG data array with shape (samples, num_groups, num_components).
        mvc_value (float or np.array): Maximum Voluntary Contraction value.
                                        If None, it will be calculated as the max across all samples.

    Returns:
        np.array: MVC-normalized EMG data.
    """
    if mvc_value is None:
        # Calculate the MVC as the maximum value across all trials and groups
        mvc_value = np.max(data)
    
    # Normalize the data by the MVC value
    normalized_data = data / mvc_value
    return normalized_data

def z_score_normalize(data):
    """
    Apply Z-score normalization to the EMG data.

    Parameters:
        data (np.array): MVC-normalized EMG data with shape (samples, num_groups, num_components).

    Returns:
        np.array: Z-score normalized EMG data.
    """
    # Calculate mean and standard deviation for each group and component
    mean = np.mean(data, axis=0)  # Mean along the time axis
    std = np.std(data, axis=0)    # Standard deviation along the time axis
    
    # Avoid division by zero
    std[std == 0] = 1
    
    # Apply Z-score normalization
    standardized_data = (data - mean) / std
    return standardized_data

def normalize_data(filtered_and_downsampled_data, mvc_value=None):
    """
    Normalize EMG data to MVC and apply Z-score normalization.

    Parameters:
        filtered_and_downsampled_data (dict): Dictionary with filtered and downsampled data.
                                              Shape: (samples, num_groups, num_components).
        mvc_value (float or None): Maximum Voluntary Contraction value. If None, it is calculated.

    Returns:
        dict: Dictionary with normalized data.
    """
    normalized_data = {}

    for subject_code, classes in filtered_and_downsampled_data.items():
        normalized_data[subject_code] = {}

        for class_label, trials in classes.items():
            normalized_data[subject_code][class_label] = []

            for trial in trials:
                # Step 1: Normalize to MVC
                mvc_normalized = normalize_to_mvc(trial, mvc_value)

                # Step 2: Apply Z-score normalization
                z_score_normalized = z_score_normalize(mvc_normalized)

                # Append the normalized trial
                normalized_data[subject_code][class_label].append(z_score_normalized)

    return normalized_data

def consolidate_groups(data):
    """
    Consolidate all trials for a class into a single array, combining the group dimension.

    Parameters:
        data (dict): Dictionary with segmented or processed data.
                     Each trial has shape (samples, num_groups, num_components).

    Returns:
        dict: Dictionary with consolidated trials for each class and participant.
              Each class will have a single array of shape (samples, total_groups, num_components).
    """
    consolidated_data = {}

    for subject_code, classes in data.items():
        consolidated_data[subject_code] = {}
        
        for class_label, trials in classes.items():
            # Flatten the group dimension across all trials and concatenate
            consolidated_trials = [
                trial.reshape(trial.shape[0], -1, trial.shape[2]) for trial in trials
            ]
            
            # Concatenate along the group axis
            consolidated_class = np.concatenate(consolidated_trials, axis=1)  # Shape: (samples, total_groups, num_components)
            
            consolidated_data[subject_code][class_label] = consolidated_class

    return consolidated_data


from imblearn.over_sampling import SMOTE
import numpy as np

def balance_classes_with_smote(consolidated_data):
    """
    Apply SMOTE to balance the classes (Baseline, Preparation, Cancellation) for each subject.

    Parameters:
        consolidated_data (dict): Dictionary containing reshaped and consolidated data.
                                  Each class per subject has shape (samples, total_groups, num_components).

    Returns:
        dict: Data with balanced classes for each subject.
    """
    balanced_data = {}

    for subject_code, classes in consolidated_data.items():
        # Flatten and combine all classes for this subject
        X = []
        y = []
        for class_label, class_data in classes.items():
            # Flatten class data: (samples, total_groups, num_components) -> (total_groups, features)
            flattened_data = class_data.transpose(1, 0, 2).reshape(-1, class_data.shape[0] * class_data.shape[2])
            X.append(flattened_data)
            y.extend([class_label] * flattened_data.shape[0])

        # Convert to numpy arrays
        X = np.vstack(X)  # Combine all samples into a single array
        y = np.array(y)

        # Encode class labels (e.g., Baseline = 0, Preparation = 1, Cancellation = 2)
        unique_labels = {label: i for i, label in enumerate(np.unique(y))}
        y_encoded = np.array([unique_labels[label] for label in y])

        # Apply SMOTE
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

        # Decode class labels back to original strings
        inverse_labels = {i: label for label, i in unique_labels.items()}
        y_resampled = [inverse_labels[label] for label in y_resampled]

        # Separate resampled data back into classes
        balanced_data[subject_code] = {}
        for class_label in classes.keys():
            # Get indices of the current class in the resampled data
            class_indices = [i for i, label in enumerate(y_resampled) if label == class_label]
            class_resampled = X_resampled[class_indices]

            # Reshape back to (samples, total_groups, num_components)
            reshaped_data = class_resampled.reshape(-1, class_data.shape[0], class_data.shape[2]).transpose(1, 0, 2)
            balanced_data[subject_code][class_label] = reshaped_data

    return balanced_data


import numpy as np
from scipy.signal import butter, filtfilt

def add_gaussian_jitter(data, noise_std=0.01):
    """
    Add Gaussian noise (jitter) to the EMG signal.

    Parameters:
        data (np.array): EMG data with shape (samples, num_groups, num_components).
        noise_std (float): Standard deviation of the Gaussian noise relative to signal amplitude.

    Returns:
        np.array: EMG data with added Gaussian noise.
    """
    max_amplitude = np.max(np.abs(data))
    noise = np.random.normal(0, noise_std * max_amplitude, data.shape)
    jittered_data = data + noise
    return jittered_data


def add_bandpass_variation(data, fs, low_cutoffs=(12, 14), high_cutoffs=(29, 31)):
    """
    Apply bandpass filter augmentation by varying the cutoff frequencies slightly.

    Parameters:
        data (np.array): EMG data with shape (samples, num_components).
        fs (int): Sampling frequency of the data (in Hz).
        low_cutoffs (tuple): Range for lower cutoff frequency.
        high_cutoffs (tuple): Range for higher cutoff frequency.

    Returns:
        np.array: EMG data with augmented bandpass filtering.
    """
    # Ensure the data is 2D (samples × components)
    if len(data.shape) != 2:
        raise ValueError(f"Expected 2D input (samples, components), got {data.shape}")

    # Randomly select cutoff frequencies
    low_cutoff = np.random.uniform(*low_cutoffs)
    high_cutoff = np.random.uniform(*high_cutoffs)

    # Design the Butterworth filter
    b, a = butter(N=4, Wn=[low_cutoff, high_cutoff], btype='band', fs=fs)

    # Apply the filter to each component
    filtered_data = np.zeros_like(data)
    for component in range(data.shape[1]):
        filtered_data[:, component] = filtfilt(b, a, data[:, component])

    return filtered_data


import random

def augment_dataset(balanced_data, augment_fraction=0.35, fs=512, noise_std=0.01, low_cutoffs=(12, 14), high_cutoffs=(29, 31)):
    """
    Augment a fraction of the dataset with mixed variations (jitter, bandpass, or both).

    Parameters:
        data (dict): Original dataset with shape (samples, num_groups, num_components).
        augment_fraction (float): Fraction of trials to augment (0 < augment_fraction <= 1).
        fs (int): Sampling frequency.
        noise_std (float): Standard deviation for Gaussian jitter.
        low_cutoffs (tuple): Lower cutoff range for bandpass variation.
        high_cutoffs (tuple): Upper cutoff range for bandpass variation.

    Returns:
        dict: Combined dataset with original and augmented trials.
    """
    augmented_data = {}

    for subject_code, classes in balanced_data.items():
        augmented_data[subject_code] = {}

        for class_label, trials in classes.items():
            augmented_trials = []
            original_trials = trials.copy()

            for trial in original_trials.transpose(1, 0, 2):
                if np.random.rand() < augment_fraction:
                    augmentation_type = random.choice(['jitter', 'bandpass', 'both'])  # Choose augmentation type

                    if augmentation_type == 'jitter':
                        augmented_trial = add_gaussian_jitter(trial, noise_std)
                    elif augmentation_type == 'bandpass':
                        augmented_trial = add_bandpass_variation(trial, fs, low_cutoffs, high_cutoffs)
                    elif augmentation_type == 'both':
                        jittered_trial = add_gaussian_jitter(trial, noise_std)
                        augmented_trial = add_bandpass_variation(jittered_trial, fs, low_cutoffs, high_cutoffs)

                    augmented_trials.append(augmented_trial)

            # Combine original and augmented trials
            if augmented_trials:
                augmented_trials_array = np.stack(augmented_trials, axis=1)
                combined_trials = np.concatenate([original_trials, augmented_trials_array], axis=1)
            else:
                combined_trials = original_trials

            augmented_data[subject_code][class_label] = combined_trials

    return augmented_data


import random

def split_by_participant(data, split_ratio=(4, 2, 1), random_seed=42):
    """
    Split the dataset into train, validation, and test sets by participant.

    Parameters:
        data (dict): Augmented data dictionary with participants and classes.
        split_ratio (tuple): Train, Val, Test ratio in no. of participants 
        random_seed (int): Seed for reproducibility of splits.

    Returns:
        tuple: Three dictionaries for train, validation, and test splits.
    """
    # Set the random seed for reproducibility
    random.seed(random_seed)

    # Shuffle participant IDs
    participants = list(data.keys())
    random.shuffle(participants)

    # Calculate split sizes
    num_participants = len(participants)
    train_size = split_ratio[0]
    val_size = split_ratio[1]
    # train_size = int(train_frac * num_participants)
    # val_size = int(val_frac * num_participants)


    # Assign participants to splits
    train_participants = participants[:train_size]
    val_participants = participants[train_size:train_size + val_size]
    test_participants = participants[train_size + val_size:]

    # Initialize dictionaries for each split
    train_data = {p: data[p] for p in train_participants}
    val_data = {p: data[p] for p in val_participants}
    test_data = {p: data[p] for p in test_participants}

    # View split ratios
    # print("Train Participants:", train_data.keys())
    # print("Validation Participants:", val_data.keys())
    # print("Test Participants:", test_data.keys())

    return train_data, val_data, test_data

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical # type: ignore
import matplotlib.pyplot as plt



# One-hot encode the labels for categorical classification
def prepare_data(data):
    """
    Prepares the train, validation, and test datasets for input into the classifier.
    
    Parameters:
        data (dict): Dictionary containing data split by participants and classes.

    Returns:
        tuple: (X, y), where X is the input array and y is the one-hot encoded labels.
    """
    X = []
    y = []

    class_mapping = {'Baseline': 0, 'Preparation': 1, 'Cancellation': 2}

    for subject, classes in data.items():
        for class_label, trials in classes.items():
            # Reshape each trial to (segment_length, num_channels) and append
            for trial in trials.transpose(1, 0, 2):  # Shape: (total_groups, samples, num_channels)
                X.append(trial)
                y.append(class_mapping[class_label])

    X = np.array(X)  # Shape: (num_samples, segment_length, num_channels)
    y = np.array(y)  # Shape: (num_samples,)
    
    # Convert labels to one-hot encoding
    y = to_categorical(y, num_classes=3)

    return X, y




