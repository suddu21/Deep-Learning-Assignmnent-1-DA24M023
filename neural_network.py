import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_sizes, output_size=10, weight_init='random', 
                 activation='sigmoid', optimizer='sgd', learning_rate=0.001, weight_decay=0, 
                 momentum=0.9, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize neural network parameters
        
        Parameters:
        input_size: int
            Number of input features
        hidden_layers: int
            Number of hidden layers
        hidden_sizes: list of int
            Number of neurons in each hidden layer
            But restricted to same size for all layers as specified in the assignment
        output_size: int
            Number of output classes
        weight_init: str
            Initialization method ('random' or 'Xavier')
        activation: str
            Activation function ('sigmoid', 'tanh', 'relu', 'identity')
        optimizer: str
            Optimization algorithm ('sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam')
        learning_rate: float
            Learning rate for optimization
        weight_decay: float
            L2 regularization parameter
        momentum: float
            Momentum parameter for momentum and nesterov optimizers
        beta, beta1, beta2: float
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
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Initialize network architecture
        self.initialize_parameters()
        
        # Initialize optimizer parameters
        self.initialize_optimizer()
        
    def initialize_parameters(self):
        """Initialize weights and biases"""

        # The weight vector will be an array of arrays
        # Each sub-array will be the weights for a layer
        # Ex: weights[0] will be the weights connecting input to first hidden layer
        # Ex: weights[1] will be the weights connecting first hidden layer to second hidden layer
        # Ex: weights[-1] will be the weights connecting last hidden layer to output layer
        # Same for biases
        self.weights, self.biases = [], []
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        for i in range(len(layer_sizes) - 1):
            if self.weight_init == 'Xavier':
                # Xavier initialization
                # This initializes weights to sqrt(6 / (n_in + n_out))
                scale = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
                # Normal Xavier initialization draws weight samples from a normal distribution
                # with limits of +/- scale
                # Reference: https://www.geeksforgeeks.org/xavier-initialization/
                w = np.random.uniform(-scale, scale, (layer_sizes[i], layer_sizes[i+1]))
            else:
                # Random initialization
                # The constant factor here makes a big difference in model performance
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            
            # All 1 bias
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def initialize_optimizer(self):
        """Initialize optimizer-specific variables"""
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        # For momentum and nesterov
        # These use only velocity vectors
        if self.optimizer in ['momentum', 'nag']:
            self.velocity_w = [np.zeros_like(w) for w in self.weights]
            self.velocity_b = [np.zeros_like(b) for b in self.biases]
        
        # For rmsprop, adam, nadam
        # These use cache to store squared gradients
        if self.optimizer in ['rmsprop', 'adam', 'nadam']:
            self.cache_w = [np.zeros_like(w) for w in self.weights]
            self.cache_b = [np.zeros_like(b) for b in self.biases]
            
        # For adam, nadam
        # These use momentum and cache for first and second moments
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
            
            # Add to the list of activations
            # Ex: A[0] is the input, A[1] is the output of the first hidden layer, etc.
            self.A.append(A)
        
        return self.A[-1]
    
    def apply_activation(self, Z):
        """Apply activation function"""
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif self.activation == 'tanh':
            return np.tanh(Z)
        elif self.activation == 'ReLU':
            return np.maximum(0, Z)
        elif self.activation == 'identity':
            return Z
        else:
            raise ValueError(f"Activation function {self.activation} not supported")
    
    def activation_derivative(self, Z):
        """Compute derivative of activation function"""
        # Referred to
        # https://dustinstansbury.github.io/theclevermachine/derivation-common-neural-network-activation-functions
        
        if self.activation == 'sigmoid':
            A = self.apply_activation(Z)
            return A * (1 - A)
        elif self.activation == 'tanh':
            return 1 - np.tanh(Z)**2
        elif self.activation == 'ReLU':
            return (Z > 0).astype(float)
        elif self.activation == 'identity':
            return np.ones_like(Z)
        else:
            raise ValueError(f"Activation function {self.activation} not supported")
    
    def softmax(self, Z):
        """Softmax activation function"""
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def compute_loss(self, y_pred, y_true, loss_fn='cross_entropy'):
        """Compute cross-entropy loss with L2 regularization"""
        m = y_true.shape[0]
        #print(y_pred.shape, y_true.shape)
        #print(y_pred[range(m), y_true])
        #print(y_pred)
        #print(y_true)

        # Cross-entropy loss
        if loss_fn == 'cross_entropy':
            log_probs = -np.log(y_pred[range(m), y_true])
            data_loss = np.sum(log_probs) / m
        elif loss_fn == 'mean_squared_error':
            # First make true label into one-hot encoded vector to subtract from predicted prob distribution
            y_true_one_hot = np.zeros_like(y_pred)
            y_true_one_hot[range(m), y_true] = 1
            # Then get mean squeared error
            data_loss = np.mean((y_pred - y_true_one_hot)**2)
        else:
            raise ValueError(f"Loss function {loss_fn} not supported")
        
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
        # Array of numpy arrays holding gradients for each layer
        # Ex: dW[0] is the gradient of the first layer weights
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Gradients for output layer
        y_one_hot = np.zeros((m, self.output_size))
        y_one_hot[range(m), y] = 1
        
        # Initial error at output layer
        dA = self.A[-1] - y_one_hot
        
        # Backpropagate through layers starting from the output layer
        for i in reversed(range(len(self.weights))):
            # First work on output layer
            if i == len(self.weights) - 1:
                # For the output layer, we already have dA computed
                dW[i] = np.dot(self.A[i].T, dA) / m
                db[i] = np.sum(dA, axis=0, keepdims=True) / m
                # Propagate error to previous layer
                dA_prev = np.dot(dA, self.weights[i].T)
            
            # Then work on hidden layers
            else:
                # For hidden layers we need to compute dZ by backpropagating through activation function
                dZ = dA_prev * self.activation_derivative(self.Z[i])
                # Compute gradients at each layer
                dW[i] = np.dot(self.A[i].T, dZ) / m
                db[i] = np.sum(dZ, axis=0, keepdims=True) / m
                # Propagate error to previous layer (except for the input layer)
                if i > 0:
                    dA_prev = np.dot(dZ, self.weights[i].T)
            
            # Add L2 regularization gradient - This is applied to weights at each layer
            if self.weight_decay > 0:
                dW[i] += self.weight_decay * self.weights[i]
        
        return dW, db
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def evaluate(self, X, y):
        """Evaluate the model"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
