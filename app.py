import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx

# Load the dataset
dataset = TUDataset(root='./data', name='ENZYMES')

# Split the dataset
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
train_size = int(len(dataset) * train_ratio)
val_size = int(len(dataset) * val_ratio)
test_size = len(dataset) - train_size - val_size

train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:train_size + val_size]
test_dataset = dataset[train_size + val_size:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Print dataset information
print(f"Total graphs: {len(dataset)}")
print(f"Train/Val/Test split: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

# Optional: Visualize one graph
graph = to_networkx(dataset[0])
nx.draw(graph, with_labels=True)
