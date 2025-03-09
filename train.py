import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import argparse
from neural_network import NeuralNetwork
from trainer import train
from data_utils import prepare_data
from experiment import run_experiment, setup_sweep

def run_single_experiment(args):
    # Run a single experiment with given parameters
    # Load and preprocess data
    x_train, y_train, X_val, y_val, x_test, y_test, classes = prepare_data(args.dataset)
    
    # Initialize wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        
    # Log hyperparameters
    wandb.config.update({
            "hidden_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "activation": args.activation,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weight_init": args.weight_init,
            "dataset": args.dataset,
            "loss_fn": args.loss_fn
        })
    
    # Create model
    hidden_sizes = [args.hidden_size] * args.num_layers
    
    model = NeuralNetwork(
        input_size=x_train.shape[1],
        hidden_layers=args.num_layers,
        hidden_sizes=hidden_sizes,
        output_size=10,
        weight_init=args.weight_init,
        activation=args.activation,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon
    )
    
    # Train model
    train_losses, train_accs, val_losses, val_accs = train(model, x_train, y_train, args.batch_size, args.epochs, X_val, y_val, loss_fn=args.loss_fn)
    
    # Evaluate on test set
    predictions = model.predict(x_test)
    test_acc = np.mean(predictions == y_test)
    #print(predictions[0], y_test[0])
    #test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # Plot confusion matrix
    test_confusion_matrix = confusion_matrix(y_test, predictions, normalize='true')
    plt.figure(figsize=(20, 15))
    ax = sns.heatmap(test_confusion_matrix, annot=test_confusion_matrix, xticklabels=classes, yticklabels=classes, cmap="Greens")
    ax.set_title("Confusion Matrix", size=16)
    ax.set_xlabel("Predicted Class", size=14)
    ax.set_ylabel("True Class", size=14)
    plt.savefig("test_confusion_matrix.png")

    wandb.log({"test_confusion_matrix": wandb.Image("test_confusion_matrix.png")})
    

    wandb.log({"test_acc": test_acc})
    wandb.finish()

def run_hyperparameter_sweep(args):
    # Bayesian hyperparameter sweep with wandb
    sweep_config = setup_sweep(args)
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
    wandb.agent(sweep_id, function=run_experiment)

def main():

    parser = argparse.ArgumentParser(description="DA6401 Assignment 1")
    
    # Single run or Sweep
    parser.add_argument("--mode", type=str, choices=["single", "sweep"], default="single",
                        help="Run a single experiment or hyperparameter sweep")
    
    # Parser for CLI arguments. It defaults to the best run's parameters
    parser.add_argument("-wp", "--wandb_project", type=str, default="fashion-mnist-nn-sweep",
                        help="WandB project name")
    parser.add_argument("-we", "--wandb_entity", type=str, default="da24m023-indian-institute-of-technology-madras",
                        help="WandB entity name")
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist",
                        help="Dataset to use")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("-l", "--loss_fn", type=str, default="cross_entropy",
                        help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str, 
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], 
                        default="nadam", help="Optimization algorithm")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.9,
                        help="Momentum parameter")
    parser.add_argument("-beta", "--beta", type=float, default=0.9,
                        help="Beta parameter for SGD with Nesterov momentum")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9,
                        help="Beta1 parameter for Adam/NAdam")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999,
                        help="Beta2 parameter for Adam/NAdam")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8,
                        help="Epsilon parameter to prevent division by zero")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0,
                        help="L2 regularization strength")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="Xavier",
                        help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=5,
                        help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128,
                        help="Number of neurons in each hidden layer")
    parser.add_argument("-a", "--activation", type=str, choices=["sigmoid", "tanh", "ReLU", "identity"], default="tanh",
                        help="Activation function")    
    # WandB options
    parser.add_argument("--wandb", action="store_false",
                        help="Enable wandb logging")
    parser.add_argument("--sweep_runs", type=int, default=100,
                        help="Number of runs for hyperparameter sweep")
    # Decided to scrap these args cos no point of not logging in wandb
    
    args = parser.parse_args()

    
    if args.mode == "single":
        args.wandb_project = "fashion-mnist-nn-single"
        run_single_experiment(args)
    else:
        if not args.wandb:
            print("Warning: Running a sweep without wandb logging. Enabling wandb.")
            args.wandb = True
        run_hyperparameter_sweep(args)

if __name__ == "__main__":
    main()
