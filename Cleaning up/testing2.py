import numpy as np
import wandb
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_sizes, output_size, weight_init='random', 
                 activation='sigmoid', optimizer='sgd', learning_rate=0.001, weight_decay=0, 
                 momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize neural network parameters
        
        Parameters:
        -----------
        input_size: int
            Number of input features
        hidden_layers: int
            Number of hidden layers
        hidden_sizes: list of int
            Number of neurons in each hidden layer
        output_size: int
            Number of output classes
        weight_init: str
            Initialization method ('random' or 'xavier')
        activation: str
            Activation function ('sigmoid', 'tanh', or 'relu')
        optimizer: str
            Optimization algorithm ('sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam')
        learning_rate: float
            Learning rate for optimization
        weight_decay: float
            L2 regularization parameter
        momentum: float
            Momentum parameter for momentum and nesterov optimizers
        beta1, beta2: float
            Parameters for adam, nadam optimizers
        epsilon: float
            Small constant to prevent division by zero
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_sizes = hidden_sizes[:hidden_layers]
        self.output_size = output_size
        self.weight_init = weight_init
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Initialize network architecture
        self.initialize_parameters()
        
        # Initialize optimizer parameters
        self.initialize_optimizer()
        
    def initialize_parameters(self):
        """Initialize weights and biases"""
        self.weights = []
        self.biases = []
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        for i in range(len(layer_sizes) - 1):
            if self.weight_init == 'xavier':
                # Xavier initialization
                scale = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
                w = np.random.uniform(-scale, scale, (layer_sizes[i], layer_sizes[i+1]))
            else:
                # Random initialization
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def initialize_optimizer(self):
        """Initialize optimizer-specific variables"""
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        # For momentum and nesterov
        if self.optimizer in ['momentum', 'nesterov']:
            self.velocity_w = [np.zeros_like(w) for w in self.weights]
            self.velocity_b = [np.zeros_like(b) for b in self.biases]
        
        # For rmsprop, adam, nadam
        if self.optimizer in ['rmsprop', 'adam', 'nadam']:
            self.cache_w = [np.zeros_like(w) for w in self.weights]
            self.cache_b = [np.zeros_like(b) for b in self.biases]
            
        # For adam, nadam
        if self.optimizer in ['adam', 'nadam']:
            self.momentum_w = [np.zeros_like(w) for w in self.weights]
            self.momentum_b = [np.zeros_like(b) for b in self.biases]
            self.t = 0
    
    def forward(self, X):
        """Forward pass through the network"""
        self.Z = []
        self.A = [X]
        
        for i in range(len(self.weights)):
            Z = np.dot(self.A[i], self.weights[i]) + self.biases[i]
            self.Z.append(Z)
            
            # Apply activation for all layers except the last (output) layer
            if i < len(self.weights) - 1:
                A = self.apply_activation(Z)
            else:
                # Softmax for output layer
                A = self.softmax(Z)
            
            self.A.append(A)
        
        return self.A[-1]
    
    def apply_activation(self, Z):
        """Apply activation function"""
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif self.activation == 'tanh':
            return np.tanh(Z)
        elif self.activation == 'relu':
            return np.maximum(0, Z)
        else:
            raise ValueError(f"Activation function {self.activation} not supported")
    
    def activation_derivative(self, Z):
        """Compute derivative of activation function"""
        if self.activation == 'sigmoid':
            A = self.apply_activation(Z)
            return A * (1 - A)
        elif self.activation == 'tanh':
            return 1 - np.tanh(Z)**2
        elif self.activation == 'relu':
            return (Z > 0).astype(float)
        else:
            raise ValueError(f"Activation function {self.activation} not supported")
    
    def softmax(self, Z):
        """Softmax activation function"""
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def compute_loss(self, y_pred, y_true):
        """Compute cross-entropy loss with L2 regularization"""
        m = y_true.shape[0]
        
        # Cross-entropy loss
        log_probs = -np.log(y_pred[range(m), y_true])
        data_loss = np.sum(log_probs) / m
        
        # L2 regularization
        reg_loss = 0
        if self.weight_decay > 0:
            for w in self.weights:
                reg_loss += 0.5 * self.weight_decay * np.sum(w**2)
        
        return data_loss + reg_loss
    
    def backward(self, X, y):
        """Backward pass (compute gradients)"""
        m = X.shape[0]
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Gradients for output layer
        y_one_hot = np.zeros((m, self.output_size))
        y_one_hot[range(m), y] = 1
        
        dA = self.A[-1] - y_one_hot  # Initial error at output layer
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                # For the output layer, we already have dA computed
                dW[i] = np.dot(self.A[i].T, dA) / m
                db[i] = np.sum(dA, axis=0, keepdims=True) / m
                # Propagate error to previous layer
                dA_prev = np.dot(dA, self.weights[i].T)
            else:
                # For hidden layers
                dZ = dA_prev * self.activation_derivative(self.Z[i])
                dW[i] = np.dot(self.A[i].T, dZ) / m
                db[i] = np.sum(dZ, axis=0, keepdims=True) / m
                # Propagate error to previous layer (except for the input layer)
                if i > 0:
                    dA_prev = np.dot(dZ, self.weights[i].T)
            
            # Add L2 regularization gradient
            if self.weight_decay > 0:
                dW[i] += self.weight_decay * self.weights[i]
        
        return dW, db
    
    def update_parameters(self, dW, db):
        """Update weights and biases based on gradients and chosen optimizer"""
        if self.optimizer == 'sgd':
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dW[i]
                self.biases[i] -= self.learning_rate * db[i]
        
        elif self.optimizer == 'momentum':
            for i in range(len(self.weights)):
                self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.learning_rate * dW[i]
                self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.learning_rate * db[i]
                
                self.weights[i] += self.velocity_w[i]
                self.biases[i] += self.velocity_b[i]
        
        elif self.optimizer == 'nesterov':
            for i in range(len(self.weights)):
                v_prev_w = self.velocity_w[i].copy()
                v_prev_b = self.velocity_b[i].copy()
                
                self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.learning_rate * dW[i]
                self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.learning_rate * db[i]
                
                self.weights[i] += -self.momentum * v_prev_w + (1 + self.momentum) * self.velocity_w[i]
                self.biases[i] += -self.momentum * v_prev_b + (1 + self.momentum) * self.velocity_b[i]
        
        elif self.optimizer == 'rmsprop':
            for i in range(len(self.weights)):
                self.cache_w[i] = self.beta2 * self.cache_w[i] + (1 - self.beta2) * (dW[i]**2)
                self.cache_b[i] = self.beta2 * self.cache_b[i] + (1 - self.beta2) * (db[i]**2)
                
                self.weights[i] -= self.learning_rate * dW[i] / (np.sqrt(self.cache_w[i]) + self.epsilon)
                self.biases[i] -= self.learning_rate * db[i] / (np.sqrt(self.cache_b[i]) + self.epsilon)
        
        elif self.optimizer == 'adam':
            self.t += 1
            
            for i in range(len(self.weights)):
                # Update biased first moment estimate
                self.momentum_w[i] = self.beta1 * self.momentum_w[i] + (1 - self.beta1) * dW[i]
                self.momentum_b[i] = self.beta1 * self.momentum_b[i] + (1 - self.beta1) * db[i]
                
                # Update biased second raw moment estimate
                self.cache_w[i] = self.beta2 * self.cache_w[i] + (1 - self.beta2) * (dW[i]**2)
                self.cache_b[i] = self.beta2 * self.cache_b[i] + (1 - self.beta2) * (db[i]**2)
                
                # Bias correction
                m_corrected_w = self.momentum_w[i] / (1 - self.beta1**self.t)
                m_corrected_b = self.momentum_b[i] / (1 - self.beta1**self.t)
                v_corrected_w = self.cache_w[i] / (1 - self.beta2**self.t)
                v_corrected_b = self.cache_b[i] / (1 - self.beta2**self.t)
                
                # Update parameters
                self.weights[i] -= self.learning_rate * m_corrected_w / (np.sqrt(v_corrected_w) + self.epsilon)
                self.biases[i] -= self.learning_rate * m_corrected_b / (np.sqrt(v_corrected_b) + self.epsilon)
        
        elif self.optimizer == 'nadam':
            self.t += 1
            
            for i in range(len(self.weights)):
                # Update biased first moment estimate
                self.momentum_w[i] = self.beta1 * self.momentum_w[i] + (1 - self.beta1) * dW[i]
                self.momentum_b[i] = self.beta1 * self.momentum_b[i] + (1 - self.beta1) * db[i]
                
                # Update biased second raw moment estimate
                self.cache_w[i] = self.beta2 * self.cache_w[i] + (1 - self.beta2) * (dW[i]**2)
                self.cache_b[i] = self.beta2 * self.cache_b[i] + (1 - self.beta2) * (db[i]**2)
                
                # Bias correction
                m_corrected_w = self.momentum_w[i] / (1 - self.beta1**self.t)
                m_corrected_b = self.momentum_b[i] / (1 - self.beta1**self.t)
                v_corrected_w = self.cache_w[i] / (1 - self.beta2**self.t)
                v_corrected_b = self.cache_b[i] / (1 - self.beta2**self.t)
                
                # Nesterov momentum
                m_bar_w = self.beta1 * m_corrected_w + (1 - self.beta1) * dW[i] / (1 - self.beta1**self.t)
                m_bar_b = self.beta1 * m_corrected_b + (1 - self.beta1) * db[i] / (1 - self.beta1**self.t)
                
                # Update parameters
                self.weights[i] -= self.learning_rate * m_bar_w / (np.sqrt(v_corrected_w) + self.epsilon)
                self.biases[i] -= self.learning_rate * m_bar_b / (np.sqrt(v_corrected_b) + self.epsilon)
    
    def train(self, X, y, batch_size, epochs, X_val=None, y_val=None):
        """Train the neural network"""
        m = X.shape[0]
        iterations = m // batch_size
        
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            correct_preds = 0
            
            for i in tqdm(range(iterations), desc=f"Epoch {epoch+1}/{epochs}"):
                # Get mini-batch
                start = i * batch_size
                end = min(start + batch_size, m)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += loss * (end - start)
                
                # Compute accuracy
                predictions = np.argmax(y_pred, axis=1)
                correct_preds += np.sum(predictions == y_batch)
                
                # Backward pass
                dW, db = self.backward(X_batch, y_batch)
                
                # Update parameters
                self.update_parameters(dW, db)
            
            # Average loss and accuracy for the epoch
            epoch_loss /= m
            epoch_acc = correct_preds / m
            
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_preds = self.predict(X_val)
                val_loss = self.compute_loss(self.forward(X_val), y_val)
                val_acc = np.mean(val_preds == y_val)
                
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
                
            # Log to wandb
            if wandb.run is not None:
                metrics = {
                    "train_loss": epoch_loss,
                    "train_acc": epoch_acc
                }
                if X_val is not None:
                    metrics.update({
                        "val_loss": val_loss,
                        "val_acc": val_acc
                    })
                wandb.log(metrics)
            config = wandb.config
            run_name = "lr_{}_act_{}_in_{}_op_{}_bs_{}_L2_{}_ep_{}_lay_{}_neur_{}".format(config.learning_rate, config.activation, config.weight_init, config.optimizer, config.batch_size, config.weight_decay, config.epochs, config.hidden_layers, config.hidden_size)
            wandb.run.name = run_name
            wandb.run.save()
        
        return train_losses, train_accs, val_losses, val_accs
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def evaluate(self, X, y):
        """Evaluate the model"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

def preprocess_data(X_train, X_test):
    """Preprocess the data: normalize and reshape"""
    # Normalize
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape (flatten)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    return X_train, X_test

def run_experiment(config=None):
    """Run experiment with given hyperparameters"""
    # Initialize wandb
    with wandb.init(config=config):
        # Get hyperparameters
        config = wandb.config
        
        # Load and preprocess the data
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_train, X_test = preprocess_data(X_train, X_test)
        
        # Split training data to create validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        
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
        model.train(X_train, y_train, config.batch_size, config.epochs, X_val, y_val)
        
        # Evaluate on test set
        test_acc = model.evaluate(X_test, y_test)
        wandb.log({"test_acc": test_acc})

def setup_sweep():
    """Set up wandb sweep configuration"""
    sweep_configuration = {
        'method': 'bayes',
        'name': 'fashion-mnist-sweep',
        'metric': {'goal': 'maximize', 'name': 'val_acc'},
        'parameters': {
            'epochs': {'values': [10, 20]},
            'hidden_layers': {'values': [3, 4, 5]},
            'hidden_size': {'values': [32, 64, 128]},
            'weight_decay': {'values': [0, 0.0005, 0.5]},
            'learning_rate': {'values': [1e-3, 1e-4]},
            'optimizer': {'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']},
            'batch_size': {'values': [16, 32, 64]},
            'weight_init': {'values': ['random', 'xavier']},
            'activation': {'values': ['sigmoid', 'tanh', 'relu']}
        }
    }
    
    return sweep_configuration

def main():
    """Main function to run the experiment"""
    # Load and preprocess data
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_test = preprocess_data(X_train, X_test)
    
    # Create validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    # If running a single experiment:
    """
    model = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_layers=3,
        hidden_sizes=[64, 64, 64],
        output_size=10,
        weight_init='xavier',
        activation='relu',
        optimizer='adam',
        learning_rate=0.001,
        weight_decay=0,
        momentum=0.9,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    )
    
    train_losses, train_accs, val_losses, val_accs = model.train(
        X_train, y_train, batch_size=32, epochs=5, X_val=X_val, y_val=y_val
    )
    
    test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    """
    
    # For hyperparameter sweep:
    sweep_config = setup_sweep()
    sweep_id = wandb.sweep(sweep_config, project="fashion-mnist-nn")
    wandb.agent(sweep_id, function=run_experiment)

if __name__ == "__main__":
    main()