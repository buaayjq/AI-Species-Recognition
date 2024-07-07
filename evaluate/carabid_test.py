from tensorflow import keras
from tensorflow.keras.layers import *
from prepare_data import *
from datetime import datetime
from tensorflow.keras.applications.inception_v3 import preprocess_input
import albumentations as A
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np
import os

class ModelTester:
    def __init__(self, model_path, dataset_path):
        self._model = keras.models.load_model(model_path)
        self._gray_aug = A.Compose([A.ToGray(p=1.0)])
        self._blur_aug = A.Compose([A.Blur(p=1.0)])
        self.standard_img = np.zeros((299, 299, 3))
        self.blurred_img = np.zeros((299, 299, 3))
        self.grayed_img = np.zeros((299, 299, 3))
        self._labels = sorted(os.listdir(dataset_path))

    def first_pass(self):
        """
        Passes a dummy input through the model to prepare it for fast predictions.
        """
        first_input = np.zeros((1, 299, 299, 3))
        self._model.predict([first_input, first_input, first_input])

    def prep_image(self, image):
        """
        Augments and preprocesses an image before it is input to a model.
        """
        standard_img = preprocess_input(image)
        blurred_img = preprocess_input(self._blur_aug(image=image)['image'])
        grayed_img = preprocess_input(self._gray_aug(image=image)['image'])
        return [np.expand_dims(standard_img, axis=0), np.expand_dims(grayed_img, axis=0), 
                np.expand_dims(blurred_img, axis=0)]

    def model_predict(self, image, model):
        """
        Uses the selected model to predict the top class for an image.
        """
        results = model.predict(image)[0]
        index = np.argmax(results) 
        value = results[index]
        label = self._labels[index]
        return label, value

    def classify(self, image_path):
        """
        Prepares an image from a selected filename and predicts its class with the model.
        """
        inputs = self.prep_image(img_to_array(load_img(image_path, target_size=(299, 299))))
        predicted_label, confidence = self.model_predict(inputs, self._model)
        return predicted_label

    def predict_single_image(self, image_path):
        """
        Predicts the class of a single image and returns the label.
        """
        inputs = self.prep_image(img_to_array(load_img(image_path, target_size=(299, 299))))
        predicted_label, _ = self.model_predict(inputs, self._model)
        return predicted_label

def get_filenames_and_labels(test_dataset):
    """
    Retrieves all filenames and their associated labels from the test dataset.
    """
    filenames = []
    labels = []
    for class_label in os.listdir(test_dataset):
        class_dir = os.path.join(test_dataset, class_label)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                filenames.append(os.path.join(class_dir, filename))
                labels.append(class_label)
    sorted_filenames_labels = sorted(zip(filenames, labels))
    filenames, labels = zip(*sorted_filenames_labels)
    return list(filenames), list(labels)

def evaluate_on_test_dataset(model_tester, test_dataset):
    """
    Evaluates the model on the test dataset, updating and printing accuracy after each prediction.
    """
    test_filenames, test_labels = get_filenames_and_labels(test_dataset)

    print("Test dataset labels (with category names):")
    for label in test_labels:
        print(f"{label}: {label}")

    correct_predictions = 0
    for i, file_path in enumerate(test_filenames):
        actual_label = test_labels[i]
        predicted_label = model_tester.predict_single_image(file_path)
        print(file_path)
        print(f"Predicted: {predicted_label}")
        print(f"Actual: {actual_label}")
        if predicted_label == actual_label:
            correct_predictions += 1
        accuracy = correct_predictions / (i + 1)
        print(f"Processed {i + 1}/{len(test_filenames)} images - Current Accuracy: {accuracy:.4f}")

dataset_path = "/home/Student/s4819764/database/carabid"
# model_path = "/home/Student/s4819764/model-saves/final/zac/savefile.hdf5"
test_dataset_path = "/home/Student/s4819764/database/carabid_test"
model_tester = ModelTester(
    model_path=f"/home/Student/s4819764/model-saves/final/ensemble_aug_Dataset.carabid_07052024-152833/classifier/savefile.hdf5",
    dataset_path=dataset_path
)

test_dataset = Dataset.carabid_test
evaluate_on_test_dataset(model_tester, test_dataset_path)
# image_path = "/home/Student/s4819764/database/carabid_test/1035167/d162s0044.jpg"
# model_tester.predict_single_image(image_path)