import os
from typing import List

import tensorflow as tf
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from keras_tuner import Hyperband
from tqdm import tqdm
from keras.src.callbacks import Callback
from keras.src.layers import BatchNormalization
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split

import config
from config import CATEGORIES, JSON_DATA, ANNS, COUNTS, DATA_DIR
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

def generator_images(objs, batch_size, mem_map=False, tf_mode=True):
    if os.path.isfile(os.path.join(DATA_DIR, "val_images.npy")):
        print("Found preprocessed dataset - loading...", end="")
        images = np.load(os.path.join(DATA_DIR, "val_images.npy"), mmap_mode="r" if mem_map else None)
        labels = np.load(os.path.join(DATA_DIR, "val_labels.npy"), mmap_mode="r" if mem_map else None)
        print("done.")
    else:
        # Your existing processing code goes here
        images, labels = [], []
        for obj in tqdm(objs, desc="Processing images"):
            filename, obj = obj
            image = load_geoimage(filename)
            images.append(np.array(image, dtype=np.uint8))
            probabilities = np.zeros(len(CATEGORIES), dtype=np.uint8)
            probabilities[list(CATEGORIES.values()).index(obj.category)] = 1
            labels.append(probabilities)

        images = np.array(images)
        labels = np.array(labels)
        print(f"Space required as uint8: {images.nbytes / 2 ** 30}GB")
        print(f"Space required before uint8: {images.nbytes / 2 ** 30}GB")

        os.makedirs(DATA_DIR, exist_ok=True)
        np.save(os.path.join(DATA_DIR, "val_images.npy"), images)
        np.save(os.path.join(DATA_DIR, "val_labels.npy"), labels)
        if mem_map:
            images = np.load(os.path.join(DATA_DIR, "val_images.npy"), mmap_mode="r")
            labels = np.load(os.path.join(DATA_DIR, "val_labels.npy"), mmap_mode="r")
        print("Saved.")

    if not tf_mode:
        raise ValueError("Pythons generators are disabled due to a bug in Python's generator detection.")
        # while True:
        #     indices = np.arange(len(images))  # Create an index array
        #
        #     num_batches_to_store = max(512 // batch_size, 1)
        #
        #     # Process data in chunks of 10 batches
        #     for chunk_start in range(0, len(images), batch_size * num_batches_to_store):
        #         chunk_end = min(chunk_start + batch_size * num_batches_to_store, len(images))
        #         chunk_indices = indices[chunk_start:chunk_end].copy()
        #
        #         # Retrieve all images for this chunk at once
        #         chunk_images = images[chunk_indices]
        #         chunk_labels = labels[chunk_indices]
        #         chunk_images = chunk_images.astype(np.float32)
        #
        #         # Yield individual batches from the chunk
        #         for i in range(0, len(chunk_images), batch_size):
        #             batch_end = min(i + batch_size, len(chunk_images))
        #             yield chunk_images[i:batch_end].astype(np.float32), chunk_labels[i:batch_end]
    else:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        # Batch the dataset
        dataset = dataset.batch(batch_size)

        # Repeat indefinitely (similar to while True:)
        dataset = dataset.repeat()

        # Prefetch for improved performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset



def augmented_generation(objs, batch_size, do_shuffle=True, sigma=10, color_noise=3, margin_noise=10, mem_map=False, tf_mode=True):
    print(DATA_DIR)
    print(os.path.join(DATA_DIR, "images.npy"))

    if os.path.isfile(os.path.join(DATA_DIR, "images.npy")):
        print("Found preprocessed dataset - loading...", end="")
        images = np.load(os.path.join(DATA_DIR, "images.npy"), mmap_mode="r" if mem_map else None)
        labels = np.load(os.path.join(DATA_DIR, "labels.npy"), mmap_mode="r" if mem_map else None)
        print("done.")
    else:
        # Your existing processing code goes here
        images, labels = [], []
        for obj in tqdm(objs, desc="Processing images"):
            filename, obj = obj
            image = load_geoimage(filename)
            images.append(np.array(image, dtype=np.uint8))
            probabilities = np.zeros(len(CATEGORIES), dtype=np.uint8)
            probabilities[list(CATEGORIES.values()).index(obj.category)] = 1
            labels.append(probabilities)

        images = np.array(images)
        labels = np.array(labels)
        print(f"Space required as uint8: {images.nbytes / 2**30}GB")
        print(f"Space required before uint8: {images.nbytes / 2**30}GB")


        os.makedirs(DATA_DIR, exist_ok=True)
        np.save(os.path.join(DATA_DIR, "images.npy"), images)
        np.save(os.path.join(DATA_DIR, "labels.npy"), labels)
        if mem_map:
            images = np.load(os.path.join(DATA_DIR, "images.npy"), mmap_mode="r")
            labels = np.load(os.path.join(DATA_DIR, "labels.npy"), mmap_mode="r")
        print("Saved.")


    if not tf_mode:
        raise ValueError("Pythons generators are disabled due to a bug in Python's generator detection.")
        # while True:
        #     indices = np.arange(len(images))  # Create an index array
        #
        #     if do_shuffle:
        #         np.random.shuffle(indices)  # Shuffle indices instead of images
        #
        #     num_batches_to_store = max(512 // batch_size, 1)
        #     (h, w) = images.shape[:2]
        #     center = (w // 2, h // 2)
        #
        #     # Process data in chunks of 10 batches
        #     for chunk_start in range(0, len(images), batch_size * num_batches_to_store):
        #         chunk_end = min(chunk_start + batch_size * num_batches_to_store, len(images))
        #         chunk_indices = indices[chunk_start:chunk_end].copy()
        #
        #         # Retrieve all images for this chunk at once
        #         chunk_images = images[chunk_indices]
        #         chunk_labels = labels[chunk_indices]
        #
        #         # Apply noise to the entire chunk if needed
        #         chunk_images += np.random.uniform(0, 2 * color_noise, size=chunk_images.shape).astype(np.uint8)
        #         chunk_images -= np.ones(chunk_images.shape, dtype=np.uint8) * color_noise
        #
        #         chunk_images = np.rot90(chunk_images, k=np.random.choice([0,1,2,3]), axes=(1, 2))
        #         chunk_images = chunk_images.astype(np.float32)
        #
        #         # Yield individual batches from the chunk
        #         for i in range(0, len(chunk_images), batch_size):
        #             batch_end = min(i + batch_size, len(chunk_images))
        #             yield chunk_images[i:batch_end].astype(np.float32), chunk_labels[i:batch_end]
    else:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        if do_shuffle:
            dataset = dataset.shuffle(buffer_size=len(images))

        def preprocess(image, label):
            # Convert the image to float32 (if not already)
            image = tf.cast(image, tf.float32)

            # Add noise: equivalent to (image + uniform(0, 2*color_noise) - color_noise)
            noise = tf.random.uniform(shape=tf.shape(image),
                                      minval=0, maxval=2 * color_noise,
                                      dtype=tf.float32)
            image = image + noise - color_noise

            # Apply a random rotation (0, 90, 180, or 270 degrees)
            k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
            image = tf.image.rot90(image, k=k)

            return image, label

        # Apply the preprocessing transformation in parallel
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        # Batch the dataset
        dataset = dataset.batch(batch_size)

        # Repeat indefinitely (similar to while True:)
        dataset = dataset.repeat()

        # Prefetch for improved performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


def train_val_split(train_batch=32, val_batch=512):
    labels = [obj.objects[0].category for obj in ANNS]
    anns_train, anns_valid = train_test_split(ANNS, test_size=0.1, random_state=1, shuffle=True, stratify=labels)
    objs_train = [(ann.filename, obj) for ann in anns_train for obj in ann.objects]
    objs_valid = [(ann.filename, obj) for ann in anns_valid for obj in ann.objects]
    # Generators
    train_generator = augmented_generation(objs_train, train_batch, do_shuffle=True)
    valid_generator = generator_images(objs_valid, val_batch)
    return train_generator, valid_generator, len(objs_train), len(objs_valid)

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
    model_name += "Out"
    # Learning rate is changed to 0.001
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'f1_score'])
    return model, model_name


def build_fcnn(hp):
    model = Sequential()
    model.add(Flatten(input_shape=(224, 224, 3)))

    # Tune number of layers
    num_layers = hp.Int('num_layers', min_value=3, max_value=3, step=1)

    for i in range(num_layers):
        # Tune number of neurons
        neurons = hp.Int(f'neurons_{i}', min_value=512, max_value=1024, step=64)

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



if __name__ == "__main__":
    import math
    import gc
    from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from keras_tuner import BayesianOptimization
    from tensorflow.keras.callbacks import ModelCheckpoint



    # Run the hyperparameter search
    batch_size = 256
    train_generator, valid_generator, sz_train, sz_val = train_val_split(train_batch=batch_size)
    train_steps = math.ceil(sz_train / batch_size)
    valid_steps = math.ceil(sz_val / batch_size)

    # Set up Bayesian Optimizer
    tuner = BayesianOptimization(
        build_fcnn,
        objective='val_accuracy',
        max_trials=60,  # Number of total trials to run
        directory='bayesian_search',
        project_name='fcnn_tuning',
        overwrite=True,
        # max_model_size=1_000_000_000,
        max_consecutive_failed_trials=2,
        executions_per_trial=1  # Disallow parallel execution
    )

    # Define callback for the search
    early_stop_tuner = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    tuner.search(
        train_generator,
        steps_per_epoch=train_steps,
        validation_data=valid_generator,
        validation_steps=valid_steps,
        epochs=25,
        callbacks=[early_stop_tuner]
    )

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    exit(0)