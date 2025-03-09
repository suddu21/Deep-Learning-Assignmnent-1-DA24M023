import numpy as np

# Reference for these are the course's previous year slides which I got from
# https://www.cse.iitm.ac.in/~miteshk/CS6910.html
# and
# https://cs231n.github.io/neural-networks-3/ (this one seems best and I used it for code implementation)
# and online blogs and articles like
# https://www.geeksforgeeks.org/optimization-techniques-for-gradient-descent/


class SGD():

    # Updates weights using the gradient of a single sample or mini-batch.
    # Formula:
    #   w_next = w - lr * grad(w)
    # where:
    # - w: Current weights
    # - lr: Learning rate
    # - grad(w): Gradient of the loss function with respect to w

    def update_parameters(self, network, dW, db):
        """Update weights and biases using SGD"""
        for i in range(len(network.weights)):
            network.weights[i] -= network.learning_rate * dW[i]
            network.biases[i] -= network.learning_rate * db[i]

class Momentum():

    # Uses a velocity term to accumulate past gradients for smoother updates.
    # Formula:
    #   v = beta * v_prev + (1 - beta) * grad(w)
    #   w_next = w - lr * v
    # where:
    # - v: Velocity term
    # - beta: Momentum factor (typically 0.9)
    # - lr: Learning rate
    # - grad(w): Gradient of the loss function

    def update_parameters(self, network, dW, db):
        """Update weights and biases using Momentum"""
        for i in range(len(network.weights)):
            network.velocity_w[i] = network.momentum * network.velocity_w[i] - network.learning_rate * dW[i]
            network.velocity_b[i] = network.momentum * network.velocity_b[i] - network.learning_rate * db[i]
            
            network.weights[i] += network.velocity_w[i]
            network.biases[i] += network.velocity_b[i]

class Nesterov():

    # Computes the gradient at a future estimate of the weights.
    # Formula:
    #   v = beta * v_prev + lr * grad(w - beta * v_prev)
    #   w_next = w - v
    # Helps reduce overshooting and provides a more adaptive step.

    def update_parameters(self, network, dW, db):
        """Update weights and biases using Nesterov momentum"""
        for i in range(len(network.weights)):
            # Save previous velocity vectors
            v_prev_w = network.velocity_w[i].copy()
            v_prev_b = network.velocity_b[i].copy()

            # Update velocity vectors            
            network.velocity_w[i] = network.momentum * network.velocity_w[i] - network.learning_rate * dW[i]
            network.velocity_b[i] = network.momentum * network.velocity_b[i] - network.learning_rate * db[i]
            
            network.weights[i] += -network.momentum * v_prev_w + (1 + network.momentum) * network.velocity_w[i]
            network.biases[i] += -network.momentum * v_prev_b + (1 + network.momentum) * network.velocity_b[i]

class RMSprop():

    # Adapts the learning rate by scaling it inversely with the square root of past gradient magnitudes.
    # Formula:
    #   s = beta * s_prev + (1 - beta) * (grad(w))**2
    #   w_next = w - lr / (sqrt(s) + epsilon) * grad(w)
    # where:
    # - s: Exponentially weighted sum of squared gradients
    # - beta: Decay factor
    # - epsilon: Small constant to prevent division by zero

    def update_parameters(self, network, dW, db):
        """Update weights and biases using RMSprop"""
        for i in range(len(network.weights)):
            network.cache_w[i] = network.beta * network.cache_w[i] + (1 - network.beta) * (dW[i]**2)
            network.cache_b[i] = network.beta * network.cache_b[i] + (1 - network.beta) * (db[i]**2)
            
            network.weights[i] -= network.learning_rate * dW[i] / (np.sqrt(network.cache_w[i]) + network.epsilon)
            network.biases[i] -= network.learning_rate * db[i] / (np.sqrt(network.cache_b[i]) + network.epsilon)

class Adam():

    # Combines momentum and RMSprop by using first and second moment estimates of the gradient.
    # Also the default optimizer in Keras hence the first one I ever used in my life without knowing what it was lol
    # Formula:
    #   m = beta1 * m_prev + (1 - beta1) * grad(w)  # First moment estimate
    #   s = beta2 * s_prev + (1 - beta2) * (grad(w))**2  # Second moment estimate
    # Bias correction:
    #   m_hat = m / (1 - beta1**step)
    #   s_hat = s / (1 - beta2**step)
    # Parameter update:
    #   w_next = w - lr / (sqrt(s_hat) + epsilon) * m_hat
    # where:
    # - beta1 (default 0.9) controls first moment decay
    # - beta2 (default 0.999) controls second moment decay

    def update_parameters(self, network, dW, db):
        """Update weights and biases using Adam"""
        network.t += 1
        
        for i in range(len(network.weights)):
            # Update biased first moment estimate
            network.momentum_w[i] = network.beta1 * network.momentum_w[i] + (1 - network.beta1) * dW[i]
            network.momentum_b[i] = network.beta1 * network.momentum_b[i] + (1 - network.beta1) * db[i]
            
            # Update biased second moment estimate
            network.cache_w[i] = network.beta2 * network.cache_w[i] + (1 - network.beta2) * (dW[i]**2)
            network.cache_b[i] = network.beta2 * network.cache_b[i] + (1 - network.beta2) * (db[i]**2)
            
            # Bias correction
            m_corrected_w = network.momentum_w[i] / (1 - network.beta1**network.t)
            m_corrected_b = network.momentum_b[i] / (1 - network.beta1**network.t)
            v_corrected_w = network.cache_w[i] / (1 - network.beta2**network.t)
            v_corrected_b = network.cache_b[i] / (1 - network.beta2**network.t)
            
            # Update parameters
            network.weights[i] -= network.learning_rate * m_corrected_w / (np.sqrt(v_corrected_w) + network.epsilon)
            network.biases[i] -= network.learning_rate * m_corrected_b / (np.sqrt(v_corrected_b) + network.epsilon)

class NAdam():

    # Improves Adam by incorporating Nesterov momentum for smoother updates.
    # Formula:
    #   m = beta1 * m_prev + (1 - beta1) * grad(w)
    # Nesterov-adjusted bias correction:
    #   m_hat = (beta1 * m + (1 - beta1) * grad(w)) / (1 - beta1**step)
    # Parameter update (same as Adam):
    #   w_next = w - lr / (sqrt(s_hat) + epsilon) * m_hat

    def update_parameters(self, network, dW, db):
        """Update weights and biases using NAdam"""
        network.t += 1
        
        for i in range(len(network.weights)):
            # Update biased first moment estimate
            network.momentum_w[i] = network.beta1 * network.momentum_w[i] + (1 - network.beta1) * dW[i]
            network.momentum_b[i] = network.beta1 * network.momentum_b[i] + (1 - network.beta1) * db[i]
            
            # Update biased second moment estimate
            network.cache_w[i] = network.beta2 * network.cache_w[i] + (1 - network.beta2) * (dW[i]**2)
            network.cache_b[i] = network.beta2 * network.cache_b[i] + (1 - network.beta2) * (db[i]**2)
            
            # Bias correction
            m_corrected_w = network.momentum_w[i] / (1 - network.beta1**network.t)
            m_corrected_b = network.momentum_b[i] / (1 - network.beta1**network.t)
            v_corrected_w = network.cache_w[i] / (1 - network.beta2**network.t)
            v_corrected_b = network.cache_b[i] / (1 - network.beta2**network.t)
            
            # Nesterov momentum
            m_bar_w = network.beta1 * m_corrected_w + (1 - network.beta1) * dW[i] / (1 - network.beta1**network.t)
            m_bar_b = network.beta1 * m_corrected_b + (1 - network.beta1) * db[i] / (1 - network.beta1**network.t)
            
            # Update parameters
            network.weights[i] -= network.learning_rate * m_bar_w / (np.sqrt(v_corrected_w) + network.epsilon)
            network.biases[i] -= network.learning_rate * m_bar_b / (np.sqrt(v_corrected_b) + network.epsilon)

def get_optimizer(name):
    """Factory function to get optimizer by name"""
    optimizers = {
        'sgd': SGD(),
        'momentum': Momentum(),
        'nag': Nesterov(),
        'rmsprop': RMSprop(),
        'adam': Adam(),
        'nadam': NAdam()
    }
    
    if name.lower() not in optimizers:
        raise ValueError(f"Optimizer {name} not supported")
    
    return optimizers[name.lower()]
