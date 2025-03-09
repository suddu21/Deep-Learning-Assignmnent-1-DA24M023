import numpy as np
# Added this to suppress TF warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Added this to suppress TF warnings
from tensorflow.keras.datasets import fashion_mnist, mnist # type: ignore
from sklearn.model_selection import train_test_split

def preprocess_data(x_train, x_test):

    # Divide by 255 to normalize pixel values
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Flatten into 784 dim vector
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    return x_train, x_test

def prepare_data(dataset="fashion_mnist"):

    # Load required dataset with it's classes
    if dataset == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        classes = [str(num) for num in range(10)]
    else:
        raise ValueError("Invalid dataset")
    
    # Data preprocessing
    x_train, x_test = preprocess_data(x_train, x_test)
    
    # Split into train and val sets
    x_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    
    return x_train, y_train, X_val, y_val, x_test, y_test, classes
