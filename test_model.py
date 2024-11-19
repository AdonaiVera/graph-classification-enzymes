import torch
from torch_geometric.loader import DataLoader
from methods.data_preprocessing import preprocess_data
from models.graph_model import evaluate
from utils.evaluator import plot_auc_test


def test_model(model_path, root, name):
    """
    Test the saved model on the dataset and calculate evaluation metrics.

    Args:
        model_path (str): Path to the saved model weights.
        root (str): Path to the dataset.
        name (str): Name of the dataset.

    Returns:
        None
    """
    # Preprocess the dataset
    _, _, test_loader, dataset = preprocess_data(root=root, name=name)

    # Load the model
    model = torch.load(model_path)
    model.eval()

    # Evaluate the model on the test set
    num_classes = dataset.num_classes
    test_acc, test_f1, auc_scores, all_labels, all_probs = evaluate(model, test_loader, num_classes)

    # Print evaluation metrics
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("\nClass-wise AUC Scores:")
    for i, auc in enumerate(auc_scores):
        print(f"Class {i}: AUC = {auc:.4f}")

    # Plot the AUC curve
    labels_one_hot = torch.nn.functional.one_hot(torch.tensor(all_labels), num_classes=num_classes).numpy()
    plot_auc_test(labels_one_hot, all_probs, num_classes, "GCN-2-Layers")


if __name__ == "__main__":
    # Path to the saved model
    model_path = "weights/gcn_2_layers.pt"

    # Dataset details
    root = "./data"
    name = "ENZYMES"

    # Test the model
    test_model(model_path, root, name)
