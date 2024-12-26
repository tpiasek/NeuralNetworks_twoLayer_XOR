import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse(k, y):
    return np.mean((k - y) ** 2)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_nodes = input_size
        self.hidden_nodes = hidden_size
        self.output_nodes = output_size
        self.weights_hidden = np.random.randn(input_size, hidden_size)
        self.weights_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.random.randn(hidden_size, 1)
        self.bias_output = np.random.randn(output_size, 1)
        self.all_error = [[] for _ in range(4)]
        self.all_loss = []
        self.hidden_layer_loss = []

    def compute_hidden(self, x):
        h = np.dot(x, self.weights_hidden) + self.bias_hidden.T
        h = sigmoid(h)
        return h

    def compute_output(self, x):
        y = np.dot(x, self.weights_output) + self.bias_output.T
        y = sigmoid(y)
        return y

    def train(self, x, y, learning_rate, iterations):
        for i in range(iterations):
            # FEEDFORWARD
            a_h = self.compute_hidden(x)  # Hidden layer activations
            a_o = self.compute_output(a_h)  # Output layer activations

            # CALCULATE ERRORS AND LOSSES
            output_loss = mse(a_o, y)  # Loss for output layer
            error = a_o - y  # Error for output layer

            # Calculate hidden layer MSE
            hidden_error = np.dot(error, self.weights_output.T) * sigmoid_derivative(np.dot(x, self.weights_hidden) + self.bias_hidden.T)
            hidden_loss = mse(hidden_error, np.zeros_like(hidden_error))  # Hidden layer target is zero in this case
            self.hidden_layer_loss.append(hidden_loss)

            # BACKPROPAGATION
            # Output layer gradients
            dZ2 = error
            dW2 = np.dot(a_h.T, error)
            db2 = np.sum(dZ2, axis=0, keepdims=True).T

            # Hidden layer gradients
            dA1 = np.dot(dZ2, self.weights_output.T)
            dZ1 = dA1 * sigmoid_derivative(np.dot(x, self.weights_hidden) + self.bias_hidden.T)
            dW1 = np.dot(x.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True).T

            # Update weights and biases
            self.weights_output -= dW2 * learning_rate
            self.bias_output -= db2 * learning_rate
            self.weights_hidden -= dW1 * learning_rate
            self.bias_hidden -= db1 * learning_rate

            # Store errors and losses
            self.all_loss.append(output_loss)
            for k in range(4):
                self.all_error[k].append(error[k][0])

            if i % 100 == 0:
                print(f"Iteration {i} - Output Loss: {output_loss}, Hidden Loss: {hidden_loss}")

            if output_loss < 0.01:
                print(f"Ending quicker after achieving given loss. [Loss: {output_loss}]")

                return error

        return error

    def predict(self, x):
        a_h = self.compute_hidden(x)
        a_o = self.compute_output(a_h)
        return a_o
