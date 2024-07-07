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
dataset_name = str(dataset).split(".")[1]
test_dataset_name = str(test_dataset).split(".")[1]
current_time = datetime.now().strftime("%d%m%Y-%H%M%S")

raw_train, raw_val = prep_dataset(dataset, 8)
train_gen, val_gen = prep_ensemble_aug_dataset(dataset, raw_train, raw_val, 8)

raw_test = prep_test_dataset(test_dataset, 8)
test_gen = prep_ensemble_aug_test_dataset(test_dataset, raw_test, 8)

# load three extractors for the classifier: one standard, one trained on grey imgs, one trained on blurred imgs
standard_model_path = f"/home/Student/s4819764/model-saves/unfiltered/extractor_Dataset.carabid_07052024-091956/extractor/savefile.hdf5"
standard_extractor = keras.models.load_model(standard_model_path).layers[0].layers[-1]
standard_extractor.trainable = False
for layer in standard_extractor.layers:
    layer._name += "_1"

gray_model_path = f"/home/Student/s4819764/model-saves/gray/extractor_Dataset.carabid_07052024-131350/extractor/savefile.hdf5"
gray_extractor = keras.models.load_model(gray_model_path).layers[0].layers[-1]
gray_extractor.trainable = False
for layer in gray_extractor.layers:
    layer._name += "_2"

blur_model_path = f"/home/Student/s4819764/model-saves/blur/extractor_Dataset.carabid_06052024-235500/extractor/savefile.hdf5"
blur_extractor = keras.models.load_model(blur_model_path).layers[0].layers[-1]
blur_extractor.trainable = False
for layer in blur_extractor.layers:
    layer._name += "_3"

# pass extracted features to an RNN for classification
concat_layer = concatenate([standard_extractor.output, gray_extractor.output, blur_extractor.output])
reshape_layer = Reshape((3, 1000), input_shape=(3000,))(concat_layer)
rnn_layer = Bidirectional(GRU(1000))(reshape_layer)
dropout_layer = Dropout(0.5)(rnn_layer)
dense_layer = Dense(1000, activation='relu')(dropout_layer)
softmax = Dense(train_gen.num_classes(), activation='softmax')(dense_layer)

classifier_model = keras.Model(inputs=[standard_extractor.input, gray_extractor.input, blur_extractor.input], outputs=softmax)
classifier_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(MetricsCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        val_pred = []
        val_true = []
        for x, y in self.validation_data:
            pred = np.argmax(self.model.predict(x), axis=-1)
            val_pred.extend(pred)
            val_true.extend(y)

        precision = precision_score(val_true, val_pred, average='macro')
        recall = recall_score(val_true, val_pred, average='macro')
        f1 = f1_score(val_true, val_pred, average='macro')

        print(f"\nEpoch {epoch + 1} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# callbacks to save the logs and model each epoch
logdir = "/home/Student/s4819764/logs/unfiltered/ensemble_aug_{0}_{1}/classifier".format(str(dataset), current_time)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model_path = "/home/Student/s4819764/model-saves/final/ensemble_aug_{0}_{1}/classifier/savefile.hdf5".format(str(dataset), current_time)
model_save_callback = keras.callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

# callback to shuffle the dataset each epoch
class ShuffleCallback(keras.callbacks.Callback):
    def __init__(self, generator):
        self._generator = generator

    def on_epoch_end(self, epoch, logs=None):
        self._generator.shuffle()

train_shuffle_callback = ShuffleCallback(train_gen)
val_shuffle_callback = ShuffleCallback(val_gen)

metrics_callback = MetricsCallback(validation_data=val_gen)
classifier_model.fit(train_gen, validation_data=val_gen, callbacks=[tensorboard_callback, model_save_callback, train_shuffle_callback, val_shuffle_callback, metrics_callback], epochs=10)

# Evaluation on test data
test_loss, test_accuracy = classifier_model.evaluate(test_gen)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Predictions on test data
test_pred = []
test_true = []

for x, y in test_gen:
    pred = np.argmax(classifier_model.predict(x), axis=-1)
    test_pred.extend(pred)
    test_true.extend(y)

# Compute metrics
test_precision = precision_score(test_true, test_pred, average='macro')
test_recall = recall_score(test_true, test_pred, average='macro')
test_f1 = f1_score(test_true, test_pred, average='macro')

print(f"\nTest Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}")

# Precision-Recall curve
test_true_bin = label_binarize(test_true, classes=range(train_gen.num_classes()))
precisions, recalls, _ = precision_recall_curve(test_true_bin.ravel(), classifier_model.predict(test_gen).ravel())

plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()

# ROC curve
plt.figure(figsize=(8, 6))
test_pred_prob = classifier_model.predict(test_gen)

for i in range(train_gen.num_classes()):
    fpr, tpr, _ = roc_curve(test_true_bin[:, i], test_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()