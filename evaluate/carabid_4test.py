from tensorflow.keras.applications import InceptionV3
from tensorflow import keras
from tensorflow.keras.layers import *
from prepare_data import *
from datetime import datetime
from tensorflow.keras.applications.inception_v3 import preprocess_input
import albumentations as A
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



dataset = Dataset.carabid
test_dataset = Dataset.carabid_test

raw_train, raw_val = prep_dataset(dataset, 8)
train_gen, val_gen = prep_ensemble_aug_dataset(dataset, raw_train, raw_val, 8)

raw_test = prep_test_dataset(test_dataset, 8)
test_gen = prep_ensemble_aug_test_dataset(test_dataset, raw_test, 8)

raw_test1, raw_test2 = prep_dataset(test_dataset, 8)
test_gen1, test_gen2 = prep_ensemble_aug_dataset(test_dataset, raw_test1, raw_test2, 8)

augment1 = A.Compose([A.Blur(p=1.0)])
augment2 = A.Compose([A.ToGray(p=1.0)])

def apply_gray(images):
    aug_imgs = []
    for img in images:
        aug_imgs.append(augment2(image=img)["image"])
    return np.array(aug_imgs)

def apply_blur(images):
    aug_imgs = []
    for img in images:
        aug_imgs.append(augment1(image=img)["image"])
    return np.array(aug_imgs)

def gray_data(images, labels):
    aug_imgs = tf.numpy_function(apply_gray, [images], tf.float32)
    return aug_imgs, labels

def blur_data(images, labels):
    aug_imgs = tf.numpy_function(apply_blur, [images], tf.float32)
    return aug_imgs, labels

extractor_gray = raw_test.map(gray_data)
extractor_blur = raw_test.map(blur_data)
# load three extractors for the classifier: one standard, one trained on grey imgs, one trained on blurred imgs
standard_model_path = f"/home/Student/s4819764/model-saves/unfiltered/extractor_Dataset.carabid_07052024-091956/extractor/savefile.hdf5"
standard_extractor = keras.models.load_model(standard_model_path)

gray_model_path = f"/home/Student/s4819764/model-saves/gray/extractor_Dataset.carabid_07052024-131350/extractor/savefile.hdf5"
gray_extractor = keras.models.load_model(gray_model_path)

blur_model_path = f"/home/Student/s4819764/model-saves/blur/extractor_Dataset.carabid_06052024-235500/extractor/savefile.hdf5"
blur_extractor = keras.models.load_model(blur_model_path)

final_model_path = f"/home/Student/s4819764/model-saves/final/ensemble_aug_Dataset.carabid_17052024-232513/classifier/savefile.hdf5"
final_extractor = keras.models.load_model(final_model_path)

# standard_loss, standard_accuracy = standard_extractor.evaluate(raw_test)
# print(f"\nstandard Loss: {standard_loss:.4f}, standard Accuracy: {standard_accuracy:.4f}")

# gray_loss, gray_accuracy = gray_extractor.evaluate(extractor_gray)
# print(f"\ngray Loss: {gray_loss:.4f}, gray Accuracy: {gray_accuracy:.4f}")

# blur_loss, blur_accuracy = blur_extractor.evaluate(extractor_blur)
# print(f"\nblur Loss: {blur_loss:.4f}, blur Accuracy: {blur_accuracy:.4f}")

final_loss, final_accuracy = final_extractor.predict(test_gen1)
print(f"\nfinal Loss: {final_loss:.4f}, final Accuracy: {final_accuracy:.4f}")

val_loss, val_accuracy = final_extractor.predict(test_gen2)
print(f"\nval Loss: {val_loss:.4f}, val Accuracy: {val_accuracy:.4f}")
