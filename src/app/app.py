import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
import matplotlib.pyplot as plt
import tempfile
import uuid
import io
import base64
from keras.src.saving import load_model
from keras_preprocessing.image import load_img, img_to_array


app = Flask(__name__)

# Load the trained model
MODEL_PATH = "../../models/chord_recognition_model.h5"
model = load_model(MODEL_PATH)

# Define parameters for spectrogram generation
IMG_HEIGHT = 128
IMG_WIDTH = 431
n_mels = 128
n_fft = 2048
hop_length = 512

# Get class names (chord names)
data_dir = "../../data/spectrograms/train"
class_names = sorted(os.listdir(data_dir))


def create_spectrogram(audio_data, sr):
    """Generate a spectrogram from audio data."""
    # Generate mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=n_fft,
                                              hop_length=hop_length, n_mels=n_mels)

    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Create the plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel')
    plt.tight_layout()
    plt.axis('off')

    # Save to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)

    # Get the image as base64 for display
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Save to temp file for model prediction
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    temp_filename = temp_file.name
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel')
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(temp_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

    return img_base64, temp_filename


def predict_chord(spectrogram_path):
    """Predict chord from spectrogram image."""
    # Load and preprocess the image
    img = load_img(spectrogram_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])

    # Get the chord name
    chord_name = class_names[predicted_class]

    # Clean up temp file
    os.unlink(spectrogram_path)

    return chord_name, confidence


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Check if the file is valid
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        # Save the uploaded file temporarily
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio.name
        file.save(temp_audio_path)

        try:
            # Load the audio file
            y, sr = librosa.load(temp_audio_path, sr=None)

            # Generate spectrogram
            spec_base64, spec_path = create_spectrogram(y, sr)

            # Make prediction
            chord_name, confidence = predict_chord(spec_path)

            # Clean up temp audio file
            os.unlink(temp_audio_path)

            return jsonify({
                'chord': chord_name,
                'confidence': f"{confidence:.2f}",
                'spectrogram': f"data:image/png;base64,{spec_base64}"
            })

        except Exception as e:
            # Clean up temp audio file
            os.unlink(temp_audio_path)
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)