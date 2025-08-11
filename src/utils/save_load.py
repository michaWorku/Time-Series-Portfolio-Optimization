"""
This script provides helper functions for saving and loading machine learning
models and data scalers. This is a crucial step for persisting trained models
and using them later for forecasting without retraining.
"""
import joblib
import os
from typing import Any
# Importing keras_load_model with an alias to avoid name clashes if other functions are added
from tensorflow.keras.models import load_model as keras_load_model


def save_pickle(obj: Any, path: str):
    """
    Saves any Python object to a file using joblib.
    This is suitable for models like ARIMA and scalers.

    Args:
        obj (Any): The Python object to save.
        path (str): The file path where the object will be saved.
    """
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    print(f"Object successfully saved to: {path}")


def load_pickle(path: str) -> Any:
    """
    Loads a Python object from a file using joblib.

    Args:
        path (str): The file path to the object.

    Returns:
        Any: The loaded Python object, or None if the file is not found.
    """
    if not os.path.exists(path):
        print(f"Error: File not found at: {path}")
        return None
    obj = joblib.load(path)
    print(f"Object successfully loaded from: {path}")
    return obj


def save_keras_model(model: Any, path: str):
    """
    Saves a trained Keras model to a file.

    Args:
        model (Any): The Keras model to save.
        path (str): The file path where the model will be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Keras model successfully saved to: {path}")


def load_keras_model(path: str):
    """
    Loads a saved Keras model from a file.

    Args:
        path (str): The file path to the model.

    Returns:
        Any: The loaded Keras model, or None if the file is not found.
    """
    if not os.path.exists(path):
        print(f"Error: Model file not found at: {path}")
        return None
    model = keras_load_model(path)
    print(f"Keras model successfully loaded from: {path}")
    return model
