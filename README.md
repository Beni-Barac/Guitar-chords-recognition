# Guitar Chord Recognition System

This project implements a neural network-based system for recognizing guitar chords from audio recordings. It uses the Osmalskyj dataset, which contains 2000 .wav samples of 10 different guitar chords.

## Features

- **Data Processing**: Partitions dataset and generates spectrograms
- **Neural Network Model**: Simple CNN architecture for audio classification
- **Web Interface**: Upload or record guitar chords for real-time recognition
- **Comprehensive Evaluation**: Provides accuracy metrics and confusion matrices

## Dataset

The Osmalskyj dataset includes:
- 2000 total samples (.wav files)
- 10 chord types with 200 samples each
- Half recorded in quiet environments, half in noisy environments
- Recorded with 4 different guitars (25 samples per guitar per condition)
- High-quality audio (44.1kHz, 16-bit)

## Project Structure

```
chord_recognition/
├── data/
│   ├── raw/                  # Original .wav files
│   ├── processed/            # Partitioned dataset
│   └── spectrograms/         # Generated spectrograms
├── models/                   # Saved model checkpoints
├── src/
│   ├── data_processing/      # Data preparation scripts
│   ├── model/                # Model architecture and training
│   ├── app/                  # Web application
│   └── main.py               # Main execution script
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/guitar-chord-recognition.git
   cd guitar-chord-recognition
   ```

2. Set up a Python virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Place the Osmalskyj dataset in the `data/raw` directory, organizing it by chord type:
   ```
   data/raw/
   ├── A_major/
   ├── C_major/
   └── ...
   ```

## Usage

### Run the Complete Pipeline

To run the entire pipeline from data processing to starting the web application:

```
python src/main.py --step all
```

### Run Individual Steps

For data partitioning:
```
python src/main.py --step partition
```

For spectrogram generation:
```
python src/main.py --step spectrograms
```

For model training:
```
python src/main.py --step train --epochs 50
```

For model evaluation:
```
python src/main.py --step evaluate
```

To start the web application:
```
python src/main.py --step app
```

## Web Interface

The web interface allows users to:
- Upload audio files of guitar chords
- Record guitar chords directly in the browser
- View the spectrogram of the audio
- See the predicted chord and confidence score

Access the web interface at `http://localhost:5000` when the application is running.

## Model Architecture

The system uses a Convolutional Neural Network (CNN) with:
- 3 convolutional blocks (32, 64, and 128 filters)
- Max pooling and batch normalization in each block
- 256 neurons in the dense layer
- Dropout (0.5) for regularization
- 10 output neurons (one per chord class)

## Evaluation

The model evaluation provides:
- Test accuracy
- Confusion matrix visualization
- Detailed classification report

## License

[Include your license information here]

## Acknowledgments

- Osmalskyj dataset creators
- [Add any other acknowledgments]