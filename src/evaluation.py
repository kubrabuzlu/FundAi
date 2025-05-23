import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import DirectoryIterator
from typing import List, Any


def evaluate_and_save_results(
    model: Model,
    test_data: DirectoryIterator,
    res_dir: str,
    model_name: str,
    dataset_name: str
) -> List[float]:
    """
    Evaluates the model on test data, saves evaluation metrics, confusion matrix,
    and classification report as image and Excel file.

    Args:
        model (Model): Trained Keras model to evaluate.
        test_data (DirectoryIterator): Test dataset iterator.
        res_dir (str): Directory path to save result files.
        model_name (str): Name of the model used in file naming.
        dataset_name (str): Name of the dataset used in file naming.

    Returns:
        List[float]: Evaluation results (loss and metrics) from model.evaluate().
    """
    # Evaluate model on test data
    results = model.evaluate(test_data)

    # Generate predictions and get predicted classes
    predictions: np.ndarray = model.predict(test_data)
    predicted_classes: np.ndarray = np.argmax(predictions, axis=1)
    true_classes: np.ndarray = test_data.classes
    class_labels: List[str] = list(test_data.class_indices.keys())

    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure()
    sns.heatmap(cm, xticklabels=class_labels, yticklabels=class_labels,
                annot=True, fmt='d', cmap='crest')
    plt.savefig(os.path.join(res_dir, f"{dataset_name}{model_name}_confusion_matrix.png"))
    plt.close()

    # Classification Report
    report: dict[str, Any] = classification_report(
        true_classes, predicted_classes,
        target_names=class_labels, output_dict=True
    )
    df_report = pd.DataFrame(report).transpose()
    df_report.to_excel(os.path.join(res_dir, f"{dataset_name}{model_name}_classification_report.xlsx"))

    # Plot Report
    df_report.iloc[:-3, :3].plot(kind='bar', rot=45)
    plt.title('Classification Report')
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, f"{dataset_name}{model_name}_classification_report.png"))
    plt.close()

    return results
