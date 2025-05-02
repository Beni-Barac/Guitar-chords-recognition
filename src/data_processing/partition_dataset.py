import os
import shutil
import random
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define splits
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def partition_dataset(source_dir, dest_dir):
    """
    Partition the dataset into train, validation, and test sets.

    Parameters:
    - source_dir: Directory containing the original dataset (organized by chord)
    - dest_dir: Directory where the partitioned dataset will be stored
    """
    # Create destination directories if they don't exist
    os.makedirs(os.path.join(dest_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'test'), exist_ok=True)

    # For each chord type
    chord_types = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for chord in chord_types:
        # Create chord directories in each split
        os.makedirs(os.path.join(dest_dir, 'train', chord), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'validation', chord), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'test', chord), exist_ok=True)

        # Get all files for this chord
        chord_files = [f for f in os.listdir(os.path.join(source_dir, chord))
                       if f.endswith('.wav')]

        # Shuffle the files
        random.shuffle(chord_files)

        # Calculate split indices
        n_files = len(chord_files)
        n_train = int(n_files * TRAIN_RATIO)
        n_val = int(n_files * VAL_RATIO)

        # Split the files
        train_files = chord_files[:n_train]
        val_files = chord_files[n_train:n_train + n_val]
        test_files = chord_files[n_train + n_val:]

        # Copy files to appropriate directories
        for f in train_files:
            shutil.copy(os.path.join(source_dir, chord, f),
                        os.path.join(dest_dir, 'train', chord, f))

        for f in val_files:
            shutil.copy(os.path.join(source_dir, chord, f),
                        os.path.join(dest_dir, 'validation', chord, f))

        for f in test_files:
            shutil.copy(os.path.join(source_dir, chord, f),
                        os.path.join(dest_dir, 'test', chord, f))

    print(f"Dataset partitioned successfully into {dest_dir}")


if __name__ == "__main__":
    # Example usage
    source_dir = "../../data/raw"
    dest_dir = "../../data/processed"
    partition_dataset(source_dir, dest_dir)