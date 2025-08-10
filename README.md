# MLP-Using-NumPy

MLP (Multi-Layer Perceptron) with two hidden layers, Adam optimizer, and optional L1/L2 regularization.
Implements forward propagation, backpropagation, and parameter updates using pure NumPy.
Designed for small-scale experiments with fully-connected feedforward neural networks.

Parameters:
    input_dim (int): Number of input features.
    h1_dim (int): Number of neurons in the first hidden layer.
    h2_dim (int): Number of neurons in the second hidden layer.
    output_dim (int): Number of output classes.
    learning_rate (float): Step size for Adam updates.
    epochs (int): Number of training iterations.
    l1_lambda (float): L1 regularization strength (default: 0.0).
    l2_lambda (float): L2 regularization strength (default: 0.0).
    beta1 (float): Adam exponential decay rate for first moment estimates.
    beta2 (float): Adam exponential decay rate for second moment estimates.
    eps (float): Adam numerical stability constant.

Methods:
    train(X, Y): Train the network on data X with labels Y.
