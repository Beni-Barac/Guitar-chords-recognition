import tensorflow as tf
from tensorflow.keras import layers, models


def create_chord_recognition_model(input_shape=(128, 431, 1), num_classes=10):
    """Create a CNN model for chord recognition.

    Parameters
    ----------
    input_shape: tuple
        Shape of the input spectrograms ``(height, width, channels)``.
    num_classes: int
        Number of chord classes to predict.

    Returns
    -------
    tensorflow.keras.Model
        A compiled Keras model ready for training.
    """

    l2_reg = tf.keras.regularizers.l2(1e-4)

    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2_reg, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Global pooling and dense layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
