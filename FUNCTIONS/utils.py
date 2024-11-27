import os
import h5py
import json
import numpy as np

# CALCULATE STATES FROM STATE_VEC
# (Originally 2-Go, 1-No Go)
def calculate_states(state_vec):
    """
    Converts the state_vec array into a condensed states list with start and end samples.
    """
    # Flatten state_vec to ensure it's 1D
    state_vec_flat = state_vec.flatten()

    # Initialize variables
    states = []
    prev_state = state_vec_flat[0]
    start_sample = 0

    # Loop through state_vec to detect state transitions
    for i, state in enumerate(state_vec_flat):
        if state != prev_state:  # State transition detected
            state_label = "GO" if prev_state == 2 else "NO-GO"
            states.append({
                'state_value': int(prev_state - 1),  # Convert 1 -> 0 (NO-GO) and 2 -> 1 (GO)
                'start': int(start_sample),
                'end': int(i - 1),
                'state': state_label
            })
            # Update for the next segment
            start_sample = i
            prev_state = state

    # Add the final segment
    state_label = "GO" if prev_state == 2 else "NO-GO"
    states.append({
        'state_value': int(prev_state - 1),
        'start': int(start_sample),
        'end': int(len(state_vec_flat) - 1),
        'state': state_label
    })

    return states

# CALCULATE STATES FROM .MAT FILE/.MAT DIRECTORY AND APPEND THEM TO MAT FILES
def append_states_to_mat_files(path):
    """
    Append calculated states to .mat files.
    
    Parameters:
    - path (str): Path to a directory or a .mat file.
    """
    # Check if the input is a directory or a single file
    if os.path.isdir(path):
        # If it's a directory, list all .mat files
        file_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.mat')]
    elif os.path.isfile(path) and path.endswith('.mat'):
        # If it's a single .mat file, create a single-item list
        file_paths = [path]
    else:
        print("Invalid path. Please provide a valid directory or a .mat file.")
        return

    # Process each .mat file
    for file_path in file_paths:
        try:
            with h5py.File(file_path, 'r+') as f:  # 'r+' mode allows read/write
                if 'state_vec' in f:
                    # Get state_vec from the file
                    state_vec = f['state_vec'][:]

                    # Calculate states (you need to define this function)
                    states = calculate_states(state_vec)

                    # Check if 'states' key already exists
                    if 'states' in f.keys():
                        print(f"'states' already exists in {file_path}. Overwriting...")
                        del f['states']  # Delete the existing key

                    # Save the states variable into the .mat file
                    print(f"Adding 'states' key to {file_path}...")
                    
                    # Save states as JSON-like string in the .mat file
                    states_json = json.dumps(states)
                    f.create_dataset('states', data=np.string_(states_json))  # Store as a JSON string
                else:
                    print(f"'state_vec' not found in {file_path}. Skipping...")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("Processing complete.")

import os
import numpy as np

def load_sets_globally(folder_path):
    """
    Load all .npy files from the given folder and declare them as global variables.

    Parameters:
        folder_path (str): Path to the folder containing .npy files.
    """
    # List to keep track of loaded variable names
    loaded_variables = []

    # Loop through all .npy files in the folder
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
            
            # Add to the list of loaded variables
            loaded_variables.append(var_name)

    # Print summary of loaded variables
    print(f"Loaded variables: {', '.join(loaded_variables)}")


