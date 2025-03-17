# [DA6401 Intro to Deep Learning - Assignment 1](https://wandb.ai/sivasankar1234/DA6401/reports/DA6401-Assignment-1--VmlldzoxMTQ2NDQwNw)
* Wandb report - https://wandb.ai/da24m023-indian-institute-of-technology-madras/fashion-mnist-nn-sweep/reports/Sudhanva-Satish-DA24M023-DA6401-Assignment-1--VmlldzoxMTY5ODY1OA?accessToken=uje0e7k9sa6p71hgv2i4q0empuhil9yzcb8wwai5e4i0fs2v3j5wlllazhzi796c
* Github repo link - https://github.com/suddu21/Deep-Learning-Assignmnent-1-DA24M023
## Overview
* The aim of this assignment is to build a neural network implementing all the operations from scratch using numpy
* The task is image classification of the fashion_mnist dataset
* Various optimization algorithms have been implemented as directed
## Code Organization
* sample_images.py - To log a sample image from each class of the dataset to wandb
* train.py - Entry file that takes the command line arguments, processes them and runs a single model run or sweep appropriately
* experiment.py - Logic for setting up a sweep experiment config and function to be executed at each sweep experiment run
* data_utils.py - To load the required dataset, preprocess it and return the train, test and validation subsets of the dataset
* trainer.py - Flow of training the model with chosen optimizer and logging results to wandb
* optimizers.py - Library of implemented optimizers
* neural_network.py - Contains the NeuralNetwork class
## Instructions to train and run the model
* **PRE-REQUISITES**
  * Login to your Wandb account before running the code
    ```
    wandb login
    ```
  * Make sure to give the correct wandb entity name and project name while running the project. Note that the default values in the code are my Wandb account values which will not work for you, so pass the right values using the CLI arguments --wandb_entity and --wandb_project
  * Install necessary libraries in requirements.txt - numpy, matplotlib, seaborn, keras, scikit-learn, wandb, argparse, tqdm, os
    ```
    pip install -r requirements.txt
    ```
* (Q1) To plot the sample images for the dataset, i.e, Question 1, run the following command
  ```
  python sample_images.py
  ```
  * This will log the sample images of both Fashion MNIST and MNIST datasets to wandb
* The driver file is train.py which can be run with the command
  
  ```
  python train.py
  ```
* It accepts all command line arguments as mentioned in the code specifications. If no arguments are passed, it defaults to the best parameters found which is detailed in the wandb report.
* For example, to run the model with tanh activation and 5 epochs

  ```
  python train.py --activation tanh --epochs 5
  ```
* When train.py is run for a single model run, it does the following:-
  1. Loads the specified dataset (fashion-mnist by default)
  2. Trains the model with parsed arguments, if any. Defaults to pre-defined hyperparams if none.
  3. Logs the run to wandb along with training and validation metrics.
  4. Evaluates the model on the test dataset and logs testing accuracy.
  5. Creates a confusion matrix for test predictions, saves it locally and logs it to wandb.
* (Q2-6) To run a hyperparameter sweep, use the CL argument --mode which takes 'single' and 'sweep' as values. Pass the value 'sweep' to run a wandb sweep.
  ```
  python train.py --mode sweep
  ```
  * This starts a Bayesian sweep across the hyperparameter search space and logs each run with appropriate name to wandb
  * To stop the sweep early, press ctrl+C in the terminal running the program or stop it from Wandb's sweep controls console
* To modify the hyperparameter search space for the sweep, go to the file experiment.py and add your choice of values in the first sweep config file of the sweep_config() method
* (Q7) To generate a confusion matrix for test data, simply run the train.py with no arguments. A confusion matrix image 'test_confusion_matrix.png' will be created, saved locally and logged to wandb.
* (Q8) To compare the model performance for MSE vs Cross Entropy loss, run the following command
  ```
  python train.py --mode sweep --loss mean_squared_error
  ```
* To run the model on the MNIST dataset, run the command
  ```
  python train.py --datatset mnist
  ```
* (Q10) To run a sweep on the MNIST dataset, run the command
  ```
  python train.py --mode sweep --dataset mnist
  ```
## Code Specifications
* These are the given command line arguments as given in the assignment. I have followed the same format however the default values I have used are based on my best params

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | *Provided in code | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | *Provided in code  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 1 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 4 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | sgd | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.1 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.5 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.5 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.5 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.5 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | random | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 1 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 4 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | sigmoid | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |

* I added support for the following additional arguments for convenience

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `--mode` | 'single' | Choices: ["single", "sweep"] |
| `-d`,`--dataset` | 'fashion-mnist' | Choices: ['fashion-mnist', 'mnist'] |
| `--sweep_runs` | `100` | Number of runs per sweep |
