import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

def analyze_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Analyze the dataset and answer key questions.

    Args:
        dataset (torch_geometric.data.Dataset): The dataset to analyze.
        train_ratio (float): Ratio of data used for training.
        val_ratio (float): Ratio of data used for validation.
        test_ratio (float): Ratio of data used for testing.

    Prints:
        - Number of data samples.
        - Problem addressed by the dataset.
        - Nodes, edges, and features explanation.
        - Missing information, if any.
        - Dataset labels.
        - Data split percentages.
        - Preprocessing steps.
    """
    # 1. Number of samples in the dataset
    num_samples = len(dataset)
    print(f"1) Number of data samples: {num_samples}")

    # 2. Problem addressed by the dataset
    print("2) Problem addressed: Graph classification.")

    # 3. Nodes, edges, and features in the dataset
    first_graph = dataset[0]
    num_nodes = first_graph.num_nodes
    num_edges = first_graph.edge_index.size(1)
    num_features = dataset.num_node_features
    print(f"3) Each graph has:\n   - Nodes: {num_nodes}\n   - Edges: {num_edges}\n   - Features per node: {num_features}")

    # 4. Missing information
    has_missing_features = first_graph.x is None
    has_missing_labels = first_graph.y is None
    if has_missing_features:
        print("4) Missing information: Node features are missing.")
    elif has_missing_labels:
        print("4) Missing information: Labels are missing.")
    else:
        print("4) No missing information detected.")

    # 5. Dataset labels
    num_classes = dataset.num_classes
    print(f"5) Labels: This is a multi-class classification problem with {num_classes} classes.")

    # 6. Data split percentages
    train_size = train_ratio * 100
    val_size = val_ratio * 100
    test_size = test_ratio * 100
    print(f"6) Data split:\n   - Training: {train_size:.1f}%\n   - Validation: {val_size:.1f}%\n   - Testing: {test_size:.1f}%")

    # 7. Preprocessing steps
    print("7) Preprocessing steps: Normalize node features and ensure all graphs are valid (e.g., no empty graphs).")

def visualize_graph(graph, node_labels=None):
    """
    Visualize a graph using NetworkX.

    Args:
        graph (torch_geometric.data.Data): Graph data.
        node_labels (str, optional): Attribute to use as node labels. Defaults to None.
    """
    G = to_networkx(graph)
    nx.draw(G, with_labels=(node_labels is not None), labels=node_labels, node_size=300, node_color="lightblue")
    plt.show()
