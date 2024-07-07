from tensorflow.keras.applications import InceptionV3
from tensorflow import keras
from tensorflow.keras.layers import *
from prepare_data import *
from datetime import datetime
from tensorflow.keras.applications.inception_v3 import preprocess_input
import albumentations as A
import tensorflow as tf

dataset = Dataset.carabid
dataset_name = str(dataset).split(".")[1]
current_time = datetime.now().strftime("%d%m%Y-%H%M%S")

augment = A.Compose([A.ToGray(p=1.0)])

def apply_aug(images):
    aug_imgs = []
    for img in images:
        aug_imgs.append(augment(image=img)["image"])
    return np.array(aug_imgs)

def process_data(images, labels):
    aug_imgs = tf.numpy_function(apply_aug, [images], tf.float32)
    return aug_imgs, labels

raw_train, raw_val = prep_dataset(dataset, 8)
extractor_train = raw_train.map(process_data)
extractor_val = raw_val.map(process_data)

# load an inception model with ImageNet weights for re-training
inception = InceptionV3(classifier_activation=None)
inception.trainable = True

# add preprocessing and augmentation to the model to improve training
inputs = keras.Input(shape=(299, 299, 3))
flip_aug = keras.layers.RandomFlip()(inputs)  # Adjusted path
rotate_aug = keras.layers.RandomRotation(0.5)(flip_aug)  # Adjusted path
preprocessing = preprocess_input(rotate_aug)
extractor = inception(preprocessing, training=False)
inception_model = keras.Model(inputs=inputs, outputs=extractor)


# callbacks to save the logs and model each epoch
extractor_logdir = "/home/Student/s4819764/logs/logs/unfiltered/extractor_{0}_{1}/extractor".format(str(dataset), current_time)
extractor_tensorboard_callback = keras.callbacks.TensorBoard(log_dir=extractor_logdir)

extractor_model_path = "/home/Student/s4819764/model-saves/gray/extractor_{0}_{1}/extractor/savefile.hdf5".format(str(dataset), current_time)
extractor_model_save_callback = keras.callbacks.ModelCheckpoint(filepath=extractor_model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

feature_extractor = keras.Sequential([inception_model, keras.layers.Dense(num_classes(dataset), activation='softmax')])
feature_extractor.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

feature_extractor.fit(extractor_train, validation_data=extractor_val, callbacks=[extractor_tensorboard_callback, extractor_model_save_callback], epochs=20)