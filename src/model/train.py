import tensorflow as tf
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .architecture import create_chord_recognition_model

# Define image dimensions
IMG_HEIGHT = 128
IMG_WIDTH = 431
BATCH_SIZE = 32


def train_model(data_dir, model_save_path, epochs=50):
    """
    Train the chord recognition model using spectrograms.

    Parameters:
    - data_dir: Directory containing spectrogram images
    - model_save_path: Path where the trained model will be saved
    - epochs: Number of training epochs

    Returns:
    - model: Trained model
    - history: Training history
    """
    # Ensure the model directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Create data generators with augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )

    # Just rescaling for validation
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale'
    )

    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(data_dir, 'validation'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale'
    )

    # Get the number of classes
    num_classes = len(train_generator.class_indices)

    # Create the model
    model = create_chord_recognition_model(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 1),
        num_classes=num_classes
    )

    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        )
    ]

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=math.ceil(train_generator.samples / BATCH_SIZE),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=math.ceil(validation_generator.samples / BATCH_SIZE),
        callbacks=callbacks
    )

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(model_save_path), 'training_history.png'))
    plt.close()

    print(f"Model trained and saved to {model_save_path}")

    return model, history


if __name__ == "__main__":
    # Example usage
    data_dir = "../../data/spectrograms"
    model_save_path = "../../models/chord_recognition_model.h5"
    train_model(data_dir, model_save_path)
