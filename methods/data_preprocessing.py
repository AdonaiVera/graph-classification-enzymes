import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

def preprocess_data(root: str, name: str, val_ratio=0.15, test_ratio=0.1, batch_size=14232, seed=11, random_seed=42):
    """
    Preprocess the dataset by splitting it into train, validation, and test sets, and creating DataLoaders.

    Args:
        root (str): Path to store the dataset.
        name (str): Name of the dataset.
        train_ratio (float): Ratio of the training set.
        val_ratio (float): Ratio of the validation set.
        test_ratio (float): Ratio of the test set.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Load the dataset
    torch.manual_seed(seed)
    dataset = TUDataset(root=root, name=name, use_node_attr=True)

    # Split dataset into training and test sets: Training 70%, test 15% and validate 15%
    temp_dataset, test_dataset = train_test_split(dataset, test_size=val_ratio, shuffle=True, random_state=random_seed)
    train_dataset, val_dataset = train_test_split(temp_dataset, test_size=test_ratio, shuffle=True, random_state=random_seed)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10)
    test_loader = DataLoader(test_dataset)
  
    return train_loader, val_loader, test_loader, dataset

