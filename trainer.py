import numpy as np
import wandb
from tqdm import tqdm
from optimizers import get_optimizer

def train(model, X, y, batch_size, epochs, X_val=None, y_val=None, loss_fn='cross_entropy'):
    """Train the neural network"""
    m = X.shape[0]
    iterations = m // batch_size
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Get the chosen optimizer
    optimizer = get_optimizer(model.optimizer)
    
    for epoch in range(epochs):
        # Shuffle the data randomly for varied results
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0
        correct_preds = 0
        
        for i in tqdm(range(iterations), desc=f"Epoch {epoch+1}/{epochs}"):
            # Create the batches for current epoch
            start = i * batch_size
            end = min(start + batch_size, m)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Forward pass to get predictions
            y_pred = model.forward(X_batch)
            
            # Compute loss with chosen loss function
            loss = model.compute_loss(y_pred, y_batch, loss_fn)

            # Add up the loss for the epoch
            epoch_loss += loss * (end - start)
            
            # Compute accuracy
            predictions = np.argmax(y_pred, axis=1)
            correct_preds += np.sum(predictions == y_batch)
            
            # Backward pass
            # This gives gradient vectors
            dW, db = model.backward(X_batch, y_batch)
            
            # Update parameters
            optimizer.update_parameters(model, dW, db)
        
        # Average loss and accuracy for the epoch
        epoch_loss /= m
        epoch_acc = correct_preds / m
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Work on validaiton set
        val_preds = model.predict(X_val)
        val_loss = model.compute_loss(model.forward(X_val), y_val, loss_fn)
        val_acc = np.mean(val_preds == y_val)
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
        # Log to wandb
        if wandb.run is not None:
            metrics = {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_acc": epoch_acc
            }
            if X_val is not None:
                metrics.update({
                    "val_loss": val_loss,
                    "val_acc": val_acc
                })
            wandb.log(metrics)
            
            # Update run name based on configuration
            config = wandb.config
            
            run_name = "lr_{}_a_{}_l_{}_wi_{}_o_{}_b_{}_wd_{}_e_{}_nhl_{}_sz_{}_d_{}".format(
                config.learning_rate, config.activation, config.loss_fn, config.weight_init,
                config.optimizer, config.batch_size, config.weight_decay, config.epochs,
                config.hidden_layers, config.hidden_size, config.dataset
            )
            """
            run_name = "lr_{}_a_{}_l_{}_wi_{}_o_{}_b_{}_e_{}_nhl_{}_sz_{}_d_{}".format(
                config.learning_rate, config.activation, config.loss_fn, config.weight_init,
                config.optimizer, config.batch_size, config.epochs,
                config.hidden_layers, config.hidden_size, config.dataset
            )"""
            wandb.run.name = run_name
            wandb.run.save()
    
    return train_losses, train_accs, val_losses, val_accs
