import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# Define image dimensions
IMG_HEIGHT = 128
IMG_WIDTH = 431
BATCH_SIZE = 32


def evaluate_model(model_path, data_dir):
    """
    Evaluate the model on the test set.

    Parameters:
    - model_path: Path to the saved model
    - data_dir: Directory containing spectrogram images

    Returns:
    - test_acc: Test accuracy
    - cm: Confusion matrix
    """
    # Load the model
    model = load_model(model_path)

    # Create test data generator
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test accuracy: {test_acc:.4f}")

    # Get predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)

    # Get true classes
    true_classes = test_generator.classes

    # Get class labels
    class_labels = list(test_generator.class_indices.keys())

    # Create confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Save the confusion matrix plot
    confusion_matrix_path = os.path.join(os.path.dirname(model_path), 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()

    # Print classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

    # Save the classification report to a file
    report_path = os.path.join(os.path.dirname(model_path), 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    return test_acc, cm


if __name__ == "__main__":
    # Example usage
    model_path = "../../models/chord_recognition_model.h5"
    data_dir = "../../data/spectrograms"
    evaluate_model(model_path, data_dir)