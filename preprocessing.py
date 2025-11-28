import os
import numpy as np
import tensorflow as tf
# ...existing code...

AUTOTUNE = tf.data.AUTOTUNE

def create_data_generators(train_dir, valid_dir, test_dir,
                           img_size=(224, 224), batch_size=32, seed=123):
    """
    Create tf.data.Dataset pipelines using image_dataset_from_directory
    (replaces ImageDataGenerator to avoid deprecation warnings).
    Returns: (train_ds, valid_ds, test_ds)
    """

    # Load datasets from directories
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Preprocessing layers
    rescale = tf.keras.layers.Rescaling(1.0 / 255.0)
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05)
    ])

    # Apply augmentation only to training, rescale for all
    train_ds = train_ds.map(lambda x, y: (rescale(augmentation(x)), y), num_parallel_calls=AUTOTUNE)
    valid_ds = valid_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)

    # Prefetch for performance

    
