import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Sigmoid i jego pochodna
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Funkcja kosztu MSE
def mse(k, y):
    return np.mean((k - y) ** 2)

# Sieć neuronowa z Momentum, adaptacyjnym współczynnikiem uczenia i mini-batch
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_nodes = input_size
        self.hidden_nodes = hidden_size
        self.output_nodes = output_size

        # Inicjalizacja wag i biasów
        self.weights_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.weights_output = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_hidden = np.random.randn(hidden_size, 1) * 0.1
        self.bias_output = np.random.randn(output_size, 1) * 0.1

        # Momentum
        self.v_w_hidden = np.zeros_like(self.weights_hidden)
        self.v_w_output = np.zeros_like(self.weights_output)
        self.v_b_hidden = np.zeros_like(self.bias_hidden)
        self.v_b_output = np.zeros_like(self.bias_output)

    def compute_hidden(self, x):
        h = np.dot(x, self.weights_hidden) + self.bias_hidden.T
        return sigmoid(h)

    def compute_output(self, x):
        y = np.dot(x, self.weights_output) + self.bias_output.T
        return sigmoid(y)

    def train(self, x, y, learning_rate, iterations, batch_size, momentum=0.9):
        for i in range(iterations):
            # Mini-batch
            indices = np.random.permutation(x.shape[0])
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for batch_start in range(0, x.shape[0], batch_size):
                x_batch = x_shuffled[batch_start:batch_start + batch_size]
                y_batch = y_shuffled[batch_start:batch_start + batch_size]

                # Feedforward
                a_h = self.compute_hidden(x_batch)
                a_o = self.compute_output(a_h)

                # Loss
                loss = mse(a_o, y_batch)
                error = a_o - y_batch

                # Backpropagation
                # Output layer gradients
                dZ2 = error
                dW2 = np.dot(a_h.T, dZ2) / batch_size
                db2 = np.sum(dZ2, axis=0, keepdims=True).T / batch_size

                # Hidden layer gradients
                dA1 = np.dot(dZ2, self.weights_output.T)
                dZ1 = dA1 * sigmoid_derivative(np.dot(x_batch, self.weights_hidden) + self.bias_hidden.T)
                dW1 = np.dot(x_batch.T, dZ1) / batch_size
                db1 = np.sum(dZ1, axis=0, keepdims=True).T / batch_size

                # Momentum update
                self.v_w_output = momentum * self.v_w_output - learning_rate * dW2
                self.v_b_output = momentum * self.v_b_output - learning_rate * db2
                self.v_w_hidden = momentum * self.v_w_hidden - learning_rate * dW1
                self.v_b_hidden = momentum * self.v_b_hidden - learning_rate * db1

                self.weights_output += self.v_w_output
                self.bias_output += self.v_b_output
                self.weights_hidden += self.v_w_hidden
                self.bias_hidden += self.v_b_hidden

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss}")

    def predict(self, x):
        a_h = self.compute_hidden(x)
        a_o = self.compute_output(a_h)
        return a_o

# Przykład problemu Titanic
# Wczytywanie danych
data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Wybór kolumn i preprocessing
data = data[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']].dropna()
data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'male' else 0)

X = data[['Pclass', 'Sex', 'Age', 'Fare']].values
y = data[['Survived']].values

# Standaryzacja danych
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trening sieci neuronowej
nn = NeuralNetwork(input_size=4, hidden_size=5, output_size=1)
nn.train(X_train, y_train, learning_rate=0.01, iterations=1000, batch_size=32)

# Predykcja i ocena
predictions = nn.predict(X_test)
predictions = (predictions > 0.5).astype(int)
accuracy = np.mean(predictions == y_test)

print(f"Accuracy on Titanic test data: {accuracy}")
