import numpy as np

class MLP:
    def __init__(self, input_dim, h1_dim, h2_dim, output_dim, learning_rate=0.001, epochs=1000, l1_lambda=0.0, l2_lambda=0.0, beta1=0.9, beta2=0.999, eps=1e-8):
        self.W1 = np.random.randn(input_dim, h1_dim).astype(np.float32) * 0.01
        self.b1 = np.zeros((1, h1_dim), dtype=np.float32)

        self.W2 = np.random.randn(h1_dim, h2_dim).astype(np.float32) * 0.01
        self.b2 = np.zeros((1, h2_dim), dtype=np.float32)

        self.W3 = np.random.randn(h2_dim, output_dim).astype(np.float32) * 0.01
        self.b3 = np.zeros((1, output_dim), dtype=np.float32)

        # Adam parameters
        self.m = {}
        self.v = {}
        for param_name in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            self.m[param_name] = np.zeros_like(getattr(self, param_name))
            self.v[param_name] = np.zeros_like(getattr(self, param_name))

        # Hyperparameters
        self.lr = learning_rate
        self.epochs = epochs
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # Adam time step

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(np.float32)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)] + 1e-9)
        return np.sum(log_likelihood) / m

    def adam_update(self, param_name, grad):
        """Adam optimizer step for one parameter."""
        self.t += 1
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)

        setattr(self, param_name, getattr(self, param_name) - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))

    def train(self, X, Y):
        m = X.shape[0]

        for epoch in range(self.epochs):
            # Forward pass
            z1 = X @ self.W1 + self.b1
            a1 = self.relu(z1)

            z2 = a1 @ self.W2 + self.b2
            a2 = self.relu(z2)

            z3 = a2 @ self.W3 + self.b3
            a3 = self.softmax(z3)

            # Loss + regularization
            loss = self.cross_entropy_loss(Y, a3)
            if self.l1_lambda > 0:
                loss += self.l1_lambda * (np.sum(np.abs(self.W1)) + np.sum(np.abs(self.W2)) + np.sum(np.abs(self.W3)))
            if self.l2_lambda > 0:
                loss += self.l2_lambda * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2) + np.sum(self.W3 ** 2))

            # Backpropagation
            dz3 = a3 - Y
            dW3 = (a2.T @ dz3) / m
            db3 = np.sum(dz3, axis=0, keepdims=True) / m

            dz2 = (dz3 @ self.W3.T) * self.relu_derivative(z2)
            dW2 = (a1.T @ dz2) / m
            db2 = np.sum(dz2, axis=0, keepdims=True) / m

            dz1 = (dz2 @ self.W2.T) * self.relu_derivative(z1)
            dW1 = (X.T @ dz1) / m
            db1 = np.sum(dz1, axis=0, keepdims=True) / m

            # Add regularization to gradients
            if self.l1_lambda > 0:
                dW1 += self.l1_lambda * np.sign(self.W1)
                dW2 += self.l1_lambda * np.sign(self.W2)
                dW3 += self.l1_lambda * np.sign(self.W3)
            if self.l2_lambda > 0:
                dW1 += 2 * self.l2_lambda * self.W1
                dW2 += 2 * self.l2_lambda * self.W2
                dW3 += 2 * self.l2_lambda * self.W3

            # Adam updates
            self.adam_update('W1', dW1)
            self.adam_update('b1', db1)
            self.adam_update('W2', dW2)
            self.adam_update('b2', db2)
            self.adam_update('W3', dW3)
            self.adam_update('b3', db3)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.4f}")
