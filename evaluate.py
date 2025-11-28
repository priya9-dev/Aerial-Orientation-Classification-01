# evaluate.py
# Evaluation metrics and visualization

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import pandas as pd


def evaluate_model(model, test_generator):
    # Reset generator for consistent predictions
    test_generator.reset()

    # Predict
    predictions = model.predict(test_generator, verbose=1)

    # Convert probabilities to class labels
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = test_generator.classes

    # Compute metrics
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes)
    recall = recall_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    print("\n========================")
    print(" MODEL EVALUATION RESULTS")
    print("========================")
    for k, v in metrics.items():
        print(f'{k}: {v:.4f}')

    # Class labels
    class_names = list(test_generator.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return metrics, cm


