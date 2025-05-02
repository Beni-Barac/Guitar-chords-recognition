import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


def create_spectrogram(audio_path, save_path, n_mels=128, n_fft=2048, hop_length=512):
    """
    Generate and save a mel-spectrogram from an audio file.

    Parameters:
    - audio_path: Path to the audio file
    - save_path: Path where the spectrogram will be saved
    - n_mels: Number of mel bands to generate
    - n_fft: Length of the FFT window
    - hop_length: Number of samples between successive frames
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)  # Use the original sample rate

    # Generate mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                              hop_length=hop_length, n_mels=n_mels)

    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Create the plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel')

    # Save as PNG with tight layout and no axes
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_dataset(data_dir, output_dir):
    """
    Process all audio files in the dataset to create spectrograms.

    Parameters:
    - data_dir: Directory containing the partitioned dataset
    - output_dir: Directory where spectrograms will be saved
    """
    # For each split (train, validation, test)
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(data_dir, split)

        # For each chord type
        for chord in os.listdir(split_dir):
            chord_dir = os.path.join(split_dir, chord)

            # Create output directory
            output_chord_dir = os.path.join(output_dir, split, chord)
            os.makedirs(output_chord_dir, exist_ok=True)

            # Process each audio file
            for audio_file in os.listdir(chord_dir):
                if audio_file.endswith('.wav'):
                    # Define paths
                    audio_path = os.path.join(chord_dir, audio_file)
                    spec_path = os.path.join(output_chord_dir,
                                             os.path.splitext(audio_file)[0] + '.png')

                    # Generate spectrogram
                    create_spectrogram(audio_path, spec_path)

    print(f"Spectrograms generated successfully in {output_dir}")


if __name__ == "__main__":
    # Example usage
    data_dir = "../../data/processed"
    output_dir = "../../data/spectrograms"
    process_dataset(data_dir, output_dir)