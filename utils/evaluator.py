import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np

def plot_auc(labels_one_hot, probs, num_classes, model_name="Model"):
    """
    Plots the AUC (ROC) curves for a model's predictions.

    Args:
        labels_one_hot (np.ndarray): One-hot encoded true labels.
        probs (np.ndarray): Predicted probabilities for each class.
        num_classes (int): Number of classes in the dataset.
        model_name (str): Name of the model (used in plot title).
    """
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_one_hot[:, i], probs[:, i])
        auc_score = roc_auc_score(labels_one_hot[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc_score:.2f})")

    # Plot diagonal
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")

    # Title and labels
    plt.title(f"ROC Curves: {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def plot_auc_test(labels_one_hot, probs, num_classes, model_name="Model"):
    """
    Plots the AUC (ROC) curves for a model's predictions.

    Args:
        labels_one_hot (np.ndarray or list): One-hot encoded true labels.
        probs (np.ndarray or list): Predicted probabilities for each class.
        num_classes (int): Number of classes in the dataset.
        model_name (str): Name of the model (used in plot title).
    """
    # Convert inputs to numpy arrays if not already
    labels_one_hot = np.array(labels_one_hot)
    probs = np.array(probs)

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        try:
            # Calculate ROC curve and AUC score for each class
            fpr, tpr, _ = roc_curve(labels_one_hot[:, i], probs[:, i])
            auc_score = roc_auc_score(labels_one_hot[:, i], probs[:, i])
            plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc_score:.2f})")
        except ValueError as e:
            print(f"Error processing class {i}: {e}")
            continue

    # Plot diagonal
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")

    # Title and labels
    plt.title(f"ROC Curves: {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
