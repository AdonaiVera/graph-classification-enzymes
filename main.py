from methods.data_preprocessing import preprocess_data
from methods.exploration_data_analysis import analyze_dataset, visualize_graph
from models.graph_model import run_experiments, run_experiments_with_learning_rates
from models.graph_attention_network import run_best_gat_experiments

if __name__ == "__main__":
    # Preprocess the dataset
    root = './data'
    name = 'ENZYMES'
    train_loader, val_loader, test_loader, dataset = preprocess_data(root=root, name=name)

    # Analyze the dataset
    analyze_dataset(dataset)

    # Visualize one graph
    visualize_graph(dataset[0])

    
    # Run experiments with different number of layers
    run_experiments(
        dataset=(train_loader, val_loader, test_loader),
        input_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
        layers=[1, 2, 5],
        hidden_channels=64,
        epochs=500
    )

    # Run experiments with different learning rates
    learning_rates = [0.001, 0.01, 0.1]
    results = run_experiments_with_learning_rates(
        dataset=(train_loader, val_loader, test_loader),
        input_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
        layers=[2], 
        hidden_channels=64,
        epochs=500,
        learning_rates=learning_rates
    )
    print("Experiment Results with Learning Rates:", results)
    

    # Run GAT experiments with the three best configurations
    results = run_best_gat_experiments(
        dataset=(train_loader, val_loader, test_loader),
        input_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
        hidden_channels=64,
        epochs=500
    )
    print("Best GAT Experiment Results:", results)
    