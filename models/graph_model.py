import torch
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import numpy as np
from utils.evaluator import plot_auc


class GCN(torch.nn.Module):
    def __init__(self, input_features, hidden_channels, num_classes, num_layers=2, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(input_features, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        # Dropout and classification layer
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
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

def run_experiments(dataset, input_features, num_classes, layers=[1, 2, 3, 4, 5], hidden_channels=64, epochs=500):
    train_loader, val_loader, test_loader = dataset

    results = {}
    for num_layers in layers:
        print(f"\nRunning experiment with {num_layers} layer(s)...")
        model = GCN(input_features, hidden_channels, num_classes, num_layers=num_layers, dropout_rate=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train and save model
        model_path = train_and_validate(model, optimizer, train_loader, val_loader, num_classes,
                                        max_epochs=epochs, model_path=f"weights/gcn_{num_layers}_layers.pt")

        # Test model
        test_acc, test_f1 = test_model(model_path, dataset, test_loader, num_classes, model_name=f"GCN-{num_layers}-Layers")

        results[num_layers] = {"accuracy": test_acc, "f1_score": test_f1}
    return results

def run_experiments_with_learning_rates(
    dataset, input_features, num_classes, layers=[2], hidden_channels=64, epochs=500, learning_rates=[0.001, 0.01, 0.1]
):
    """
    Runs experiments for a fixed architecture with multiple learning rates.

    Args:
        dataset (tuple): (train_loader, val_loader, test_loader)
        input_features (int): Number of input features per node.
        num_classes (int): Number of output classes.
        layers (list): List of layer configurations to test (default is [2]).
        hidden_channels (int): Number of hidden neurons.
        epochs (int): Number of epochs for training.
        learning_rates (list): List of learning rates to try.

    Returns:
        dict: Results for each learning rate.
    """
    train_loader, val_loader, test_loader = dataset

    results = {}
    for lr in learning_rates:
        print(f"\nTesting with Learning Rate: {lr}")
        for num_layers in layers:
            print(f"Running experiment with {num_layers} layer(s)...")

            # Create the model
            model = GCN(input_features, hidden_channels, num_classes, num_layers=num_layers, dropout_rate=0.5)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Train and save model
            model_path = train_and_validate(
                model, optimizer, train_loader, val_loader, num_classes, max_epochs=epochs, model_path=f"weights/gcn_{num_layers}_layers_lr_{lr}.pt"
            )

            # Test model
            test_acc, test_f1 = test_model(
                model_path, dataset, test_loader, num_classes, model_name=f"GCN-{num_layers}-Layers-LR-{lr}"
            )

            # Store results
            results[f"{num_layers}_layers_lr_{lr}"] = {"accuracy": test_acc, "f1_score": test_f1}

    return results
