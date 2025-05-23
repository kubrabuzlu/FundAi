import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import DirectoryIterator
from typing import Tuple

def create_train_test_datasets(data_dir: str, img_size: Tuple[int, int], batch_size: int, seed: int) -> Tuple[DirectoryIterator, DirectoryIterator]:
    """
    Creates training and testing datasets using ImageDataGenerator.

    Args:
        data_dir (str): Root directory for the dataset.
        img_size (Tuple[int, int]): Target image size (height, width).
        batch_size (int): Number of samples per batch.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[DirectoryIterator, DirectoryIterator]: Train and test dataset iterators.
    """
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        vertical_flip=True,
        horizontal_flip=True
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    train_data = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=seed
    )

    test_data = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_data, test_data
