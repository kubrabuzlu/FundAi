import tensorflow as tf
from tensorflow.keras import Model
from typing import Tuple


def load_base_model(class_num: int, input_shape: Tuple[int, int, int], model_name: str) -> tf.keras.Model:
    """
    Loads a pretrained base model from Keras applications.

    Args:
        class_num (int): Number of output classes.
        input_shape (Tuple[int, int, int]): Shape of the input image.
        model_name (str): Name of the model architecture to load.

    Returns:
        tf.keras.Model: Pretrained base model.
    """
    model_zoo = {
        "MobileNetV2": tf.keras.applications.MobileNetV2,
        "DenseNet121": tf.keras.applications.DenseNet121,
        "InceptionV3": tf.keras.applications.InceptionV3,
        "Xception": tf.keras.applications.Xception,
    }

    if model_name not in model_zoo:
        raise ValueError(f"Unsupported model: {model_name}")

    base_model = model_zoo[model_name](include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = True

    return base_model


def create_model(class_num: int, input_shape: Tuple[int, int, int], model_name: str) -> tf.keras.Model:
    """
    Builds and compiles the complete model using the selected base model.

    Args:
        class_num (int): Number of output classes.
        input_shape (Tuple[int, int, int]): Input image shape.
        model_name (str): Base model architecture name.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    base_model = load_base_model(class_num, input_shape, model_name)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(class_num, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-4) -> tf.keras.Model:
    """
    Compiles the model with Adam optimizer and categorical crossentropy.

    Args:
        model (tf.keras.Model): The model to compile.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tf.keras.Model: Compiled model.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
