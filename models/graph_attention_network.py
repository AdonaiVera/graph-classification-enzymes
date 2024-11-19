import torch
from torch_geometric.nn import GATConv, BatchNorm, global_mean_pool
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from utils.evaluator import plot_auc


class GAT(torch.nn.Module):
    def __init__(self, input_features, hidden_channels, num_classes, num_layers=2, heads=1, dropout_rate=0.5):
        """
        Graph Attention Network (GAT) for graph classification.

        Args:
            input_features (int): Number of input features per node.
            hidden_channels (int): Number of hidden neurons in each layer.
            num_classes (int): Number of output classes for classification.
            num_layers (int): Number of GAT layers.
            heads (int): Number of attention heads for GAT layers.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Input GAT layer
        self.convs.append(GATConv(input_features, hidden_channels, heads=heads, dropout=dropout_rate))
        self.bns.append(BatchNorm(hidden_channels * heads))

        # Hidden GAT layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout_rate))
            self.bns.append(BatchNorm(hidden_channels * heads))

        # Dropout and classification layer
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.fc = torch.nn.Linear(hidden_channels * heads, num_classes)

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the GAT model.

        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Graph connectivity (edge indices).
            batch (torch.Tensor): Batch vector to identify graph components.

        Returns:
            torch.Tensor: Class predictions for each graph.
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = self.dropout(x)

        # Pooling and classification
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


def train(model, optimizer, loader):
    model.train()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()

        # Compute metrics
        total_loss += loss.item()
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
        correct += (preds == data.y).sum().item()
        total += data.y.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, accuracy, f1


@torch.no_grad()
def evaluate(model, loader, num_classes):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        prob = F.softmax(out, dim=1)
        preds = prob.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
        all_probs.extend(prob.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="weighted")

    # Class-wise AUC scores
    labels_one_hot = np.eye(num_classes)[all_labels]
    auc_scores = [roc_auc_score(labels_one_hot[:, i], np.array(all_probs)[:, i]) for i in range(num_classes)]

    return accuracy, f1, auc_scores, all_labels, all_probs


def train_and_validate(model, optimizer, train_loader, val_loader, num_classes, max_epochs=100, model_path="model.pt"):
    train_losses, val_accuracies, val_f1s = [], [], []

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc, train_f1 = train(model, optimizer, train_loader)
        val_acc, val_f1, _, _, _ = evaluate(model, val_loader, num_classes)

        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    # Save the model
    torch.save(model, model_path)
    end_time = time.time()
    print(f"Training complete in {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
    return model_path


def test_model(model_path, dataset, test_loader, num_classes, model_name="Model"):
    # Load model
    model = torch.load(model_path)
    model.eval()

    # Evaluate
    test_acc, test_f1, auc_scores, all_labels, all_probs = evaluate(model, test_loader, num_classes)

    # Print metrics
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}\n")
    print("Class-wise AUC Scores:")
    for i, auc in enumerate(auc_scores):
        print(f"Class {i}: AUC = {auc:.4f}")

    # Plot AUC curve
    labels_one_hot = np.eye(num_classes)[all_labels]  # One-hot encode true labels
    plot_auc(labels_one_hot, np.array(all_probs), num_classes, model_name)

    return test_acc, test_f1


def run_best_gat_experiments(dataset, input_features, num_classes, hidden_channels=64, epochs=500):
    """
    Runs experiments for the three best GAT configurations.

    Args:
        dataset (tuple): (train_loader, val_loader, test_loader).
        input_features (int): Number of input features per node.
        num_classes (int): Number of output classes.
        hidden_channels (int): Number of hidden neurons.
        epochs (int): Number of training epochs.

    Returns:
        dict: Results for the three best configurations.
    """
    train_loader, val_loader, test_loader = dataset
    best_configs = [(2, 2), (3, 4)]  # (layers, heads)
    results = {}

    for layers, heads in best_configs:
        print(f"\nRunning experiment with {layers} layers and {heads} attention heads...")

        model = GAT(
            input_features=input_features,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            num_layers=layers,
            heads=heads,
            dropout_rate=0.5
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train and save model
        model_path = train_and_validate(
            model, optimizer, train_loader, val_loader, num_classes, max_epochs=epochs, model_path=f"weights/gat_{layers}_layers_{heads}_heads.pt"
        )

        # Test model
        test_acc, test_f1 = test_model(
            model_path, dataset, test_loader, num_classes, model_name=f"GAT-{layers}-Layers-{heads}-Heads"
        )

        results[f"{layers}_layers_{heads}_heads"] = {"accuracy": test_acc, "f1_score": test_f1}

    return results
