import numpy as np
import matplotlib.pyplot as plt

#loss functions
def cross_entropy_loss(y_true, y_pred):
    samples_amount = y_true.shape[0]
    #y_pred_clipped = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.sum(y_true * np.log(y_pred)) / samples_amount

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class Network:
    def __init__(self, layers, activations, loss_function="mse", seed=None):
        if seed:
            np.random.seed(seed)  
        
        self.num_layers = len(layers)
        self.weights = []
        self.biases = []
        self.activations = activations  
        self.loss_function = loss_function
        
        #reate weights and biases
        for i in range(self.num_layers - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]))
            self.biases.append(np.zeros((1, layers[i + 1])))

        self.weight_error_history = []
        self.bias_error_history = []

    #activation functions
    def apply_activation(self, z, activation):
        if activation == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif activation == "relu":
            return np.maximum(0, z)
        elif activation == "linear":
            return z

    #derivatives
    def apply_activation_derivative(self, a, activation):
        if activation == "sigmoid":
            return a * (1 - a)
        elif activation == "relu":
            return np.where(a > 0, 1, 0)
        elif activation == "linear":
            return 1

    #forward
    def forward(self, X):
        self.activations_values = [X]
        self.z_values = []
        
        for i in range(self.num_layers - 1):
            z = np.dot(self.activations_values[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.apply_activation(z, self.activations[i])
            self.activations_values.append(a)
        
        return self.activations_values[-1]

    #backward
    def backward(self, X, y, output, learning_rate):
        deltas = []

        output_error = y - output  
        output_delta = output_error * self.apply_activation_derivative(output, self.activations[-1])

        deltas.append(output_delta)
        
        #backpropagation
        for i in range(self.num_layers - 2, 0, -1):
            z = self.z_values[i - 1]  
            delta = deltas[-1].dot(self.weights[i].T) * self.apply_activation_derivative(self.activations_values[i], self.activations[i - 1])
            deltas.append(delta)

        deltas.reverse()  

        #update weights and biases
        for i in range(self.num_layers - 1):
            weight_update = self.activations_values[i].T.dot(deltas[i]) * learning_rate
            bias_update = np.sum(deltas[i], axis=0, keepdims=True) * learning_rate
            
            self.weights[i] += weight_update
            self.biases[i] += bias_update
            
            #save the errors
            self.weight_error_history.append(np.linalg.norm(weight_update))
            self.bias_error_history.append(np.linalg.norm(bias_update))
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):

            output = self.forward(X)
            
            self.backward(X, y, output, learning_rate)
            
            #loss at every 1000 epochs
            if epoch % 1000 == 0:
                if self.loss_function == "cross_entropy":
                    loss = cross_entropy_loss(y, output)
                else:
                    loss = mean_squared_error(y, output)
                print(f'Epoch {epoch}, Loss: {loss}')

    # Function to plot weight and bias error history
    def plot_error_history(self):
        weight_error_history = np.array(self.weight_error_history)
        bias_error_history = np.array(self.bias_error_history)

        # Plotting
        fig, axs = plt.subplots(2, figsize=(10, 8))

        # Plot weight updates
        axs[0].plot(weight_error_history, label='Weight Update Norms')
        axs[0].set_title('Weight Error Over Epochs')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Error (Norm)')
        axs[0].legend()

        # Plot bias updates
        axs[1].plot(bias_error_history, label='Bias Update Norms')
        axs[1].set_title('Bias Error Over Epochs')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Error (Norm)')
        axs[1].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    #XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    #parameters
    layers = [2, 4, 4, 1]
    activations = ["relu", "relu", "sigmoid"]  #"relu", "sigmoid" or "linear"
    learning_rate = 0.05
    epochs = 10000
    seed = 42
    loss_function = "cross_entropy"  #"cross_entropy" or "mse"

    #initialize and train the network
    nn = Network(layers, activations, loss_function=loss_function, seed=seed)
    nn.train(X, y, epochs, learning_rate)

    #plot the error history
    nn.plot_error_history()