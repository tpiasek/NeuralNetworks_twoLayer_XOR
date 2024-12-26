import numpy as np
import matplotlib.pyplot as plt
from NeuralNetworkBase import NeuralNetwork


# _____DATA_____
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])


# _____NEURAL-NETWORK_____
nn_xor = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

# Neural network training
nn_xor.train(X, Y, 0.1, 10000)

# Final estimation for each input
for i in range(X.shape[0]):
    print(f"Input: {X[i]}  |  Predict: {nn_xor.predict(X[i])}  |  Actual: {Y[i]}")


# _____PLOTS_____
_, ax1 = plt.subplots(1, 3, figsize=(15, 5))

# Output loss
ax1[0].plot(range(len(nn_xor.all_loss)), nn_xor.all_loss, 'b', label='Output Loss')
ax1[0].set_xlabel('Iterations')
ax1[0].set_ylabel('Loss')
ax1[0].set_title('Loss through learning')
ax1[0].legend()

# Loss for each input
colors = "rbmy"
for k in range(len(colors)):
    ax1[1].plot(range(len(nn_xor.all_error[k])), nn_xor.all_error[k], colors[k], label=f"{X[k]}")
ax1[1].set_xlabel('Iterations')
ax1[1].set_ylabel('Error')
ax1[1].set_title('Error through learning')
ax1[1].legend(loc='upper right')

# Hidden layer loss
ax1[2].plot(range(len(nn_xor.hidden_layer_loss)), nn_xor.hidden_layer_loss, 'g', label='Hidden Layer Loss')
ax1[2].set_xlabel('Iterations')
ax1[2].set_ylabel('Hidden Layer MSE')
ax1[2].set_title('Hidden Layer Loss through learning')
ax1[2].legend()

plt.show()
