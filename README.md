# [Deep Learning Assignmnent 1 DA24M023](https://wandb.ai/sivasankar1234/DA6401/reports/DA6401-Assignment-1--VmlldzoxMTQ2NDQwNw?target=_blank)
## Instructions
* The goal of this assignment is twofold: (i) implement and use gradient descent (and its variants) with backpropagation for a classification task (ii) get familiar with Wandb which is a cool tool for running and keeping track of a large number of experiments.
* This is a individual assignment and no groups are allowed.
* Collaborations and discussions with other students is strictly prohibited.
* You must use Python (NumPy and Pandas) for your implementation.
* You cannot use the following packages from Keras, PyTorch, Tensorflow: optimizers, layers
* If you are using any packages from Keras, PyTorch, Tensorflow then post on Moodle first to check with the instructor.
* You have to generate the report in the same format as shown below using wandb.ai. You can start by cloning this report using the clone option above. Most of the plots that we have asked for below can be (automatically) generated using the APIs provided by wandb.ai. You will upload a link to this report on Gradescope.
* You also need to provide a link to your GitHub code as shown below. Follow good software engineering practices and set up a GitHub repo for the project on Day 1. Please do not write all code on your local machine and push everything to GitHub on the last day. The commits in GitHub should reflect how the code has evolved during the course of the assignment.
* You have to check Moodle regularly for updates regarding the assignment.

## Problem Statement
In this assignment you need to implement a feedforward neural network and write the backpropagation code for training the network. We strongly recommend using numpy for all matrix/vector operations. You are not allowed to use any automatic differentiation packages. This network will be trained and tested using the Fashion-MNIST dataset. Specifically, given an input image (28 x 28 = 784 pixels) from the Fashion-MNIST dataset, the network will be trained to classify the image into 1 of 10 classes.

Your code will have to follow the format specified in the Code Specifications section.

 ## Tasks
- [ ] Question 1 (2 Marks) - Download the fashion-MNIST dataset and plot 1 sample image for each class as shown in the grid below. Use from keras.datasets import fashion_mnist for getting the fashion mnist dataset.
- [ ] Question 2 (10 Marks) - Implement a feedforward neural network which takes images from the fashion-mnist data as input and outputs a probability distribution over the 10 classes. Your code should be flexible such that it is easy to change the number of hidden layers and the number of neurons in each hidden layer.
- [ ] Question 3 (24 Marks) - Implement the backpropagation algorithm with support for the following optimisation functions - sgd, momentum based gradient descent, nesterov accelerated gradient descent, rmsprop, adam, nadam (12 marks for the backpropagation framework and 2 marks for each of the optimisation algorithms above). We will check the code for implementation and ease of use (e.g., how easy it is to add a new optimisation algorithm such as Eve). Note that the code should be flexible enough to work with different batch sizes.
