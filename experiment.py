import wandb
from neural_network import NeuralNetwork
from trainer import train
from data_utils import prepare_data

def run_experiment(config=None):
    """Run experiment with given hyperparameters"""
    # Initialize wandb
    with wandb.init(config=config):
        # Get chisen hyperparameters
        config = wandb.config
        
        # Load up the dataset
        X_train, y_train, X_val, y_val, X_test, y_test, classes = prepare_data(dataset=config.dataset)
        #print("Loaded data")
        
        # Create neural network with specified hyperparameters
        hidden_sizes = [config.hidden_size] * config.hidden_layers
        
        # Initialize the model
        model = NeuralNetwork(
            input_size=X_train.shape[1],
            hidden_layers=config.hidden_layers,
            hidden_sizes=hidden_sizes,
            output_size=10,
            weight_init=config.weight_init,
            activation=config.activation,
            optimizer=config.optimizer,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8
        )
        
        # Train the model
        train(model, X_train, y_train, config.batch_size, config.epochs, X_val, y_val)
        
        # Evaluate on test set
        test_acc = model.evaluate(X_test, y_test)
        wandb.log({"test_acc": test_acc})

def setup_sweep(args):
    """Set up wandb sweep configuration"""
    sweep_configuration = {
        'method': 'bayes',
        'name': 'fashion-mnist-sweep',
        'entity': args.wandb_entity,
        'project': args.wandb_project,
        'metric': {'goal': 'maximize', 'name': 'val_acc'},
        'parameters': {
            'epochs': {'values': [5, 10]},
            'hidden_layers': {'values': [3, 4, 5]},
            'hidden_size': {'values': [64, 128, 256]},
            'weight_decay': {'values': [0, 0.0005, 0.5]},
            'learning_rate': {'values': [1e-3, 1e-4]},
            'momentum': {'values': [0.9, 0.5]},
            'beta': {'values': [0.9, 0.5]},
            'beta1': {'values': [0.9, 0.5]},
            'beta2': {'values': [0.999]},
            'epsilon': {'values': [1e-8]},
            'optimizer': {'values': ['momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
            'batch_size': {'values': [32, 64, 128]},
            'weight_init': {'values': ['random', 'Xavier']},
            'activation': {'values': ['tanh', 'ReLU']},
            'loss_fn': {'values': [args.loss_fn]},
            'dataset': {'values': [args.dataset]}
        }
    }

    # For Cross entrop vs MSE comparison, comment out for regular use
    if args.loss_fn == "mean_squared_error":
        sweep_configuration = {
            'method': 'grid',
            'name': 'mse vs cross entropy test',
            'entity': args.wandb_entity,
            'project': args.wandb_project,
            'metric': {'goal': 'maximize', 'name': 'val_acc'},
            'parameters': {
                'epochs': {'values': [10]},
                'hidden_layers': {'values': [3]},
                'hidden_size': {'values': [64]},
                'weight_decay': {'values': [0.0005]},
                'learning_rate': {'values': [1e-3]},
                'momentum': {'values': [0.9]},
                'beta': {'values': [0.9]},
                'beta1': {'values': [0.9]},
                'beta2': {'values': [0.999]},
                'epsilon': {'values': [1e-8]},
                'optimizer': {'values': ['rmsprop']},
                'batch_size': {'values': [32]},
                'weight_init': {'values': ['random']},
                'activation': {'values': ['tanh']},
                'loss_fn': {'values': ['cross_entropy', 'mean_squared_error']},
                'dataset': {'values': [args.dataset]}
                }
            }
    # For MNIST dataset
    if args.dataset == "mnist":

        sweep_configuration = {
        'method': 'bayes',
        'name': 'mnist-sweep',
        'entity': args.wandb_entity,
        'project': args.wandb_project,
        'metric': {'goal': 'maximize', 'name': 'val_acc'},
        'parameters': {
            'epochs': {'values': [5, 10]},
            'hidden_layers': {'values': [3, 4, 5]},
            'hidden_size': {'values': [64, 128, 256]},
            'weight_decay': {'values': [0, 0.0005, 0.5]},
            'learning_rate': {'values': [1e-3, 1e-4]},
            'momentum': {'values': [0.9, 0.5]},
            'beta': {'values': [0.9, 0.5]},
            'beta1': {'values': [0.9, 0.5]},
            'beta2': {'values': [0.999]},
            'epsilon': {'values': [1e-8]},
            'optimizer': {'values': ['momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
            'batch_size': {'values': [32, 64, 128]},
            'weight_init': {'values': ['random', 'Xavier']},
            'activation': {'values': ['tanh', 'ReLU']},
            'loss_fn': {'values': [args.loss_fn]},
            'dataset': {'values': ['mnist']}
            }
            }
    
    return sweep_configuration
