import os
import random
from typing import List

import tensorflow as tf
from keras_tuner import Hyperband
from tqdm import tqdm
from keras.src.callbacks import Callback
from keras.src.layers import BatchNormalization
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split

from config import CATEGORIES, JSON_DATA, ANNS, COUNTS
import cv2
import json
import uuid
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import warnings
import zipfile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from keras import initializers, regularizers


class GenericObject:
    """
    Generic object data.
    """
    def __init__(self):
        self.id = uuid.uuid4()
        self.bb = (-1, -1, -1, -1)
        self.category= -1
        self.score = -1

class GenericImage:
    """
    Generic image data.
    """
    def __init__(self, filename):
        self.filename = filename
        self.tile = np.array([-1, -1, -1, -1])  # (pt_x, pt_y, pt_x+width, pt_y+height)
        self.objects = list([])

    def add_object(self, obj: GenericObject):
        self.objects.append(obj)



for json_img, json_ann in zip(JSON_DATA['images'].values(), JSON_DATA['annotations'].values()):
    image = GenericImage(json_img['filename'])
    image.tile = np.array([0, 0, json_img['width'], json_img['height']])
    obj = GenericObject()
    obj.bb = (
    int(json_ann['bbox'][0]), int(json_ann['bbox'][1]), int(json_ann['bbox'][2]), int(json_ann['bbox'][3]))
    obj.category = json_ann['category_id']
    # Resampling strategy to reduce training time
    COUNTS[obj.category] += 1
    image.add_object(obj)
    ANNS.append(image)
print(COUNTS)


def load_geoimage(filename):
    warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
    src_raster = rasterio.open('xview_recognition/'+filename, 'r')
    # RasterIO to OpenCV (see inconsistencies between libjpeg and libjpeg-turbo)
    input_type = src_raster.profile['dtype']
    input_channels = src_raster.count
    img = np.zeros((src_raster.height, src_raster.width, src_raster.count), dtype=input_type)
    for band in range(input_channels):
        img[:, :, band] = src_raster.read(band+1)
    return img

def generator_images(objs, batch_size, do_shuffle=False):
    while True:
        if do_shuffle:
            np.random.shuffle(objs)
        groups = [objs[i:i+batch_size] for i in range(0, len(objs), batch_size)]
        for group in groups:
            images, labels = [], []
            for (filename, obj) in group:
                # Load image
                images.append(load_geoimage(filename))
                probabilities = np.zeros(len(CATEGORIES))
                probabilities[list(CATEGORIES.values()).index(obj.category)] = 1
                labels.append(probabilities)
            images = np.array(images).astype(np.float32)
            labels = np.array(labels).astype(np.float32)
            yield images, labels


def augmented_generation(objs, batch_size, do_shuffle=True, repeats=1, sigma = 10, tf_mode=False):
    if os.path.exists("images.npy"):
        print("Found preprocessed dataset - loading...", end="")
        images = np.load("images.npy", mmap_mode="r")
        labels = np.load("labels.npy", mmap_mode="r")
        print("done.")
    else:
        # Your existing processing code goes here
        images, labels = [], []
        for obj in tqdm(objs, desc="Processing images"):
            filename, obj = obj
            image = load_geoimage(filename)
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)

            for k in range(repeats):
                for angle in [0, 90, 180, 270]:
                    if repeats > 1:
                        # Rotate by a small random angle (e.g., between -2 and 2 degrees)
                        angle = angle + random.uniform(-5, 5)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        t_image = cv2.warpAffine(image, M, (w, h))

                        # Crop a small margin from the edges (adjust the margin as needed)
                        margin = int(round(random.uniform(0, 10)))  # number of pixels to crop from each side
                        t_image = t_image[margin:h - margin, margin:w - margin, :]
                        t_image = cv2.resize(t_image, (w, h), interpolation=cv2.INTER_LINEAR)
                    else:
                        t_image = image
                    images.append(t_image)
                    probabilities = np.zeros(len(CATEGORIES))
                    probabilities[list(CATEGORIES.values()).index(obj.category)] = 1
                    labels.append(probabilities)

        images = np.array(images)
        labels = np.array(labels).astype(np.float32)

        np.save("images.npy", images)
        np.save("labels.npy", labels)
        images = np.load("images.npy", mmap_mode="r")
        labels = np.load("labels.npy", mmap_mode="r")
        print("Saved.")

    if not tf_mode:
        # images = images.astype(np.float32)
        # np.save("images.npy", images)
        # exit(0)

        while True:
            indices = np.arange(len(images))  # Create an index array

            if do_shuffle:
                np.random.shuffle(indices)  # Shuffle indices instead of images

            num_batches_to_store = 4

            # Process data in chunks of 10 batches
            for chunk_start in range(0, len(images), batch_size * num_batches_to_store):
                chunk_end = min(chunk_start + batch_size * num_batches_to_store, len(images))
                chunk_indices = indices[chunk_start:chunk_end].copy()

                # Retrieve all images for this chunk at once
                chunk_images = images[chunk_indices].astype(np.float32)
                chunk_labels = labels[chunk_indices]

                # Apply noise to the entire chunk if needed
                if sigma > 0:
                    noise = np.random.normal(loc=0, scale=sigma, size=chunk_images.shape)
                    chunk_images += noise

                # Yield individual batches from the chunk
                for i in range(0, len(chunk_indices), batch_size):
                    batch_end = min(i + batch_size, len(chunk_indices))
                    yield chunk_images[i:batch_end], chunk_labels[i:batch_end]
    else:
        buffer_size = 10000
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        # Add shuffling if requested
        if do_shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        # Add noise function if needed
        if sigma > 0:
            def add_noise(image, label):
                noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=sigma, dtype=image.dtype)
                return image + noise, label

            dataset = dataset.map(add_noise, num_parallel_calls=tf.data.AUTOTUNE)

        # Batch the data
        dataset = dataset.batch(batch_size)

        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


def train_val_split(batch_size=32, generated=True):
    labels = [obj.objects[0].category for obj in ANNS]
    anns_train, anns_valid = train_test_split(ANNS, test_size=0.1, random_state=1, shuffle=True, stratify=labels)
    objs_train = [(ann.filename, obj) for ann in anns_train for obj in ann.objects]
    objs_valid = [(ann.filename, obj) for ann in anns_valid for obj in ann.objects]
    # Generators
    if generated:
        train_generator = augmented_generation(objs_train, batch_size, do_shuffle=True)
    else:
        train_generator = generator_images(objs_train, batch_size, do_shuffle=True)
    valid_generator = generator_images(objs_valid, batch_size, do_shuffle=False)
    return train_generator, valid_generator, len(objs_train), len(objs_valid)
# def augment_image(image, crop_ratio=0.8):
#     """
#     Augments a single image by applying rotations and center crops.
#
#     Parameters:
#         image (np.array): The input image array.
#         crop_ratio (float): The fraction of the image to keep when cropping (default 0.8).
#
#     Returns:
#         List[np.array]: A list of augmented image arrays.
#     """
#     augmented = []
#     # Apply rotations: 0, 90, 180, 270 degrees.
#     for k in range(4):
#         # Rotate the image by 90 degrees k times.
#         rotated = np.rot90(image, k=k)
#         augmented.append(rotated)
#
#         # Compute dimensions for center crop.
#         h, w = rotated.shape[:2]
#         crop_h, crop_w = int(crop_ratio * h), int(crop_ratio * w)
#         start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
#         cropped = rotated[start_y:start_y + crop_h, start_x:start_x + crop_w]
#         augmented.append(cropped)
#     return augmented
#
#
# def augment_batch(images, labels, crop_ratio=0.8):
#     """
#     Augments a batch of images and duplicates the labels accordingly.
#
#     Parameters:
#         images (List[np.array]): List or array of images.
#         labels (List[np.array]): List or array of corresponding labels.
#         crop_ratio (float): The fraction of the image to keep when cropping.
#
#     Returns:
#         Tuple[np.array, np.array]: Augmented images and labels.
#     """
#     aug_images = []
#     aug_labels = []
#     for img, lab in zip(images, labels):
#         augmented_imgs = augment_image(img, crop_ratio)
#         aug_images.extend(augmented_imgs)
#         aug_labels.extend([lab] * len(augmented_imgs))
#     return np.array(aug_images, dtype=np.float32), np.array(aug_labels, dtype=np.float32)



def draw_confusion_matrix(cm, categories):
    # Draw confusion matrix
    fig = plt.figure(figsize=[6.4*pow(len(categories), 0.5), 4.8*pow(len(categories), 0.5)])
    ax = fig.add_subplot(111)
    cm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], np.finfo(np.float64).eps)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=list(categories.values()), yticklabels=list(categories.values()), ylabel='Annotation', xlabel='Prediction')
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=int(20-pow(len(categories), 0.5)))
    fig.tight_layout()
    plt.show(fig)


def generate_predictions(model, out_name="predictions"):
    model.load_weights('model.keras', by_name=True)
    predictions_data = {"images": {}, "annotations": {}}
    for idx, ann in enumerate(ANNS):
        image_data = {"image_id": ann.filename.split('/')[-1], "filename": ann.filename, "width": int(ann.tile[2]),
                      "height": int(ann.tile[3])}
        predictions_data["images"][idx] = image_data
        # Load image
        image = load_geoimage(ann.filename)
        for obj_pred in ann.objects:
            # Generate prediction
            warped_image = np.expand_dims(image, 0)
            predictions = model.predict(warped_image, verbose=0)
            # Save prediction
            pred_category = list(CATEGORIES.values())[np.argmax(predictions)]
            pred_score = np.max(predictions)
            annotation_data = {"image_id": ann.filename.split('/')[-1], "category_id": pred_category,
                               "bbox": [int(x) for x in obj_pred.bb]}
            predictions_data["annotations"][idx] = annotation_data

    with open(f"{out_name}.json", "w") as outfile:
        json.dump(predictions_data, outfile)

    zip_filename = f"{out_name}.zip"
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(f"{out_name}.json")


class LayerSettings:
    def __init__(self, neurons=512, penalization=None, activation='elu', initialization=None, dropout=0, batch_normalization=True):
        self.neurons = neurons
        self.penalization = penalization
        self.activation = activation
        self.initialization = ('glorot' if activation in ['sigmoid', 'tanh'] else 'he') if initialization is None else initialization
        self.batch_normalization = batch_normalization
        self.dropout = dropout


# Deprecated function
def generate_fcnn_model(layers : List[LayerSettings]=(),
                        optimizer=Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=True, clipnorm=1.0)):
    model = Sequential()

    model.add(Flatten(input_shape=(224, 224, 3)))
    model_name = "ffnn_Flatten_"

    for settings in layers:
        model_name += f"_L({settings.neurons}-Reg{settings.penalization}-{settings.activation}-{settings.initialization}-Drop{settings.dropout}-BN{settings.batch_normalization})_"
        if settings.initialization.lower() == 'he':
            init = initializers.HeNormal()
        elif settings.initialization.lower() == 'glorot':
            init = initializers.GlorotUniform()
        else:
            init = initializers.RandomNormal()
        # Set regularizer if penalization is provided.
        if settings.penalization is not None:
            name, phi = settings.penalization
            if name == "l2":
                reg = regularizers.l2(phi)
            elif name == "l1":
                reg = regularizers.l1(phi)
            elif name == "l1_l2":
                l1_value, l2_value = phi if isinstance(phi, tuple) else (phi, phi)
                reg = regularizers.l1_l2(l1=l1_value, l2=l2_value)
            else:
                raise ValueError(f"Unknown regularization type: {name}")
        else:
            reg = None

        activation = Activation(settings.activation)

        model.add(Dense(settings.neurons, kernel_initializer=init, kernel_regularizer=reg))
        if settings.batch_normalization:
            model.add(BatchNormalization())
        model.add(activation)
        if settings.dropout > 0:
            model.add(Dropout(settings.dropout))

    model.add(Dense(len(CATEGORIES), activation='softmax'))
    model_name = "Out"
    # Learning rate is changed to 0.001
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'f1_score'])
    return model, model_name

def build_fcnn(hp):
    # First, define a model-building function for Hyperband
    model = Sequential()
    model.add(Flatten(input_shape=(224, 224, 3)))

    # Tune number of layers
    num_layers = hp.Int('num_layers', min_value=3, max_value=4, step=1)

    for i in range(num_layers):
        # Tune number of neurons
        neurons = hp.Int(f'neurons_{i}', min_value=512, max_value=1024, step=128)

        # Tune regularization
        l1_value = hp.Float(f'l1_{i}', min_value=1e-4, max_value=1e-1, sampling='log')
        l2_value = hp.Float(f'l2_{i}', min_value=1e-4, max_value=1e-1, sampling='log')

        if l1_value > 0 and l2_value == 0:
            l1_value = hp.Float(f'l1_{i}', min_value=1e-4, max_value=1e-1, sampling='log')
            reg = regularizers.l1(l1_value)
        elif l1_value == 0 and l2_value > 0:
            l2_value = hp.Float(f'l2_{i}', min_value=1e-4, max_value=1e-1, sampling='log')
            reg = regularizers.l2(l2_value)
        elif l1_value > 0 and l2_value > 0:
            reg = regularizers.l1_l2(l1=l1_value, l2=l2_value)

        # Tune activation function
        activation = hp.Choice(f'activation_{i}', values=['leaky_relu', 'elu'])

        # Tune initialization
        init_type = hp.Choice(f'init_{i}', values=['he', 'glorot'])
        if init_type == 'he':
            init = initializers.HeNormal()
        elif init_type == 'glorot':
            init = initializers.GlorotUniform()
        else:
            init = initializers.RandomNormal()

        # Add Dense layer with the selected parameters
        model.add(Dense(neurons, kernel_initializer=init, kernel_regularizer=reg))

        # Tune batch normalization
        use_bn = hp.Boolean(f'batch_norm_{i}')
        if use_bn:
            model.add(BatchNormalization())

        model.add(Activation(activation))

        # Tune dropout rate
        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.4, step=0.1)
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(len(CATEGORIES), activation='softmax'))

    # Tune learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    # Create optimizer with tunable parameters
    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=hp.Float('beta_1', min_value=0.8, max_value=0.999, default=0.9),
        beta_2=hp.Float('beta_2', min_value=0.8, max_value=0.999, default=0.999),
        epsilon=1e-8,
        amsgrad=hp.Boolean('amsgrad', default=True),
        clipnorm=hp.Float('clipnorm', min_value=0.5, max_value=2.0, default=1.0)
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'f1_score']
    )

    return model





import time


class TimingCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.start_time
        print(f"\nEpoch {epoch + 1} took {duration:.2f} seconds\n")


class HistorySaverCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if not hasattr(self, 'history_data'):
            self.history_data = []
        self.history_data.append(logs)

    def on_train_end(self, logs=None):
        # Save the history to a JSON file at the end of training
        with open('training_history.json', 'w') as f:
            json.dump(self.history_data, f)


class HyperbandCheckpointCallback(Callback):
    """
    Custom callback to save the best hyperparameters found so far during Hyperband search.
    """

    def __init__(self, tuner: Hyperband, save_dir='hyperband_checkpoints'):
        super().__init__()
        self.tuner = tuner
        self.save_dir = save_dir
        self.best_val_accuracy = 0
        self.trial_count = 0

        # Create the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def on_epoch_end(self, epoch, logs=None):
        """Save the current best hyperparameters at the end of each epoch"""
        if logs and 'val_accuracy' in logs:
            current_val_accuracy = logs['val_accuracy']

            # If this is a new best model
            if current_val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = current_val_accuracy

                # Get the current trial's hyperparameters
                hyperparameters = self.tuner.get_best_hyperparameters()

                # Create a dict with all relevant information
                checkpoint_data = {
                    'val_accuracy': current_val_accuracy,
                    'epoch': epoch,
                    'hyperparameters': hyperparameters
                }

                # Save to file
                filename = f"trial_acc_{current_val_accuracy}_epoch_{epoch}.json"
                checkpoint_file = os.path.join(self.save_dir, filename)
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

                print(f"\nNew best hyperparameters found! Val accuracy: {current_val_accuracy:.4f}")
                print(f"Saved to {checkpoint_file}")

    def on_train_begin(self, logs=None):
        """Log the start of a new trial"""
        self.trial_count += 1
        print(f"\nStarting trial {self.trial_count}")

if __name__ == "__main__":
    import math

    layers = [
        LayerSettings(512, ('l2', 0.01), 'relu', 'he', 0.2, True),
        LayerSettings(512, ('l2', 0.01), 'relu', 'he', 0.2, True),
        LayerSettings(512, ('l2', 0.01), 'relu', 'he', 0.2, True)
    ]
    model, name = generate_fcnn_model(layers)

    from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    model_checkpoint = ModelCheckpoint(f'{name}.keras', monitor='val_accuracy', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=10, verbose=1)
    early_stop = EarlyStopping('val_accuracy', patience=40, verbose=1)
    terminate = TerminateOnNaN()
    callbacks = [model_checkpoint, reduce_lr, early_stop, terminate]

    callbacks = callbacks + [TimingCallback(), HistorySaverCallback()]

    batch_size = 256
    train_generator, valid_generator, sz_train, sz_val = train_val_split(batch_size=batch_size)
    epochs = 80
    train_steps = math.ceil(sz_train / batch_size)
    valid_steps = math.ceil(sz_val / batch_size)
    h = model.fit(train_generator,
                  steps_per_epoch=train_steps,
                  validation_data=valid_generator,
                  validation_steps=valid_steps,
                  epochs=epochs,
                  callbacks=callbacks,
                  verbose=1,
                  )
    # Best validation model
    best_idx = int(np.argmax(h.history['val_accuracy']))
    best_value = np.max(h.history['val_accuracy'])
    print('Best validation model: epoch ' + str(best_idx + 1), ' - val_accuracy ' + str(best_value))