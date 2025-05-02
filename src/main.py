import os
import argparse
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data_processing.partition_dataset import partition_dataset
from src.data_processing.generate_spectrograms import process_dataset
from src.model.train import train_model
from src.model.evaluate import evaluate_model


def setup_dirs():
    """Create necessary directories for the project."""
    os.makedirs(os.path.join(project_root, 'data', 'raw'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'data', 'processed'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'data', 'spectrograms'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Guitar Chord Recognition Pipeline')
    parser.add_argument('--step', choices=['all', 'partition', 'spectrograms', 'train', 'evaluate', 'app'],
                        default='all', help='Step to execute')
    parser.add_argument('--raw-dir', default=os.path.join(project_root, 'data', 'raw'),
                        help='Directory containing raw audio files')
    parser.add_argument('--processed-dir', default=os.path.join(project_root, 'data', 'processed'),
                        help='Directory where processed dataset will be stored')
    parser.add_argument('--spec-dir', default=os.path.join(project_root, 'data', 'spectrograms'),
                        help='Directory where spectrograms will be stored')
    parser.add_argument('--model-path', default=os.path.join(project_root, 'models', 'chord_recognition_model.h5'),
                        help='Path where model will be saved')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')

    args = parser.parse_args()

    # Create project directories
    setup_dirs()

    if args.step in ['all', 'partition']:
        print("Step 1: Partitioning dataset...")
        partition_dataset(args.raw_dir, args.processed_dir)

    if args.step in ['all', 'spectrograms']:
        print("Step 2: Generating spectrograms...")
        process_dataset(args.processed_dir, args.spec_dir)

    if args.step in ['all', 'train']:
        print("Step 3: Training model...")
        train_model(args.spec_dir, args.model_path, epochs=args.epochs)

    if args.step in ['all', 'evaluate']:
        print("Step 4: Evaluating model...")
        test_acc, _ = evaluate_model(args.model_path, args.spec_dir)
        print(f"Final test accuracy: {test_acc:.4f}")

    if args.step in ['all', 'app']:
        print("Step 5: Starting web application...")
        # Import here to avoid loading Flask when not needed
        from src.app.app import app
        app.run(debug=True)


if __name__ == "__main__":
    main()