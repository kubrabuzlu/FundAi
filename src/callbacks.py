from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Tuple


def get_callbacks(patience_es: int = 10, patience_lr: int = 5) -> Tuple[EarlyStopping, ReduceLROnPlateau]:
    """
    Creates callback functions for early stopping and learning rate reduction.

    Args:
        patience_es (int): Patience for early stopping.
        patience_lr (int): Patience for learning rate reduction.

    Returns:
        Tuple[EarlyStopping, ReduceLROnPlateau]: Configured callback functions.
    """
    early_stop = EarlyStopping(monitor='val_loss', patience=patience_es, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_lr, verbose=1)
    return early_stop, reduce_lr
