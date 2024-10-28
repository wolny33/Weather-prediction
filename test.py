import matplotlib
import matplotlib.pyplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loss functions
def cross_entropy_loss(y_true, y_pred):
    samples_amount = y_true.shape[0]
    y_pred_clipped = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.sum(y_true * np.log(y_pred_clipped)) / samples_amount

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

        # Initialize weights and biases
        for i in range(self.num_layers - 1):
            limit = np.sqrt(6 / (layers[i] + layers[i + 1])) 
            self.weights.append(np.random.uniform(-limit, limit, (layers[i], layers[i + 1])))
            self.biases.append(np.zeros((1, layers[i + 1])))

        self.weight_error_history = []
        self.bias_error_history = []

    def apply_activation(self, z, activation):
        if activation == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif activation == "relu":
            return np.maximum(0, z)
        elif activation == "linear":
            return z
        elif activation == "cube":
            return z**3
        elif activation == "softmax":
            #print("Input to softmax (z):", z)
            #print("Max value in z:", np.max(z))
            #print("Min value in z:", np.min(z))
            z_stable = z - np.max(z, axis=1, keepdims=True)
            exps = np.exp(z_stable)
            return exps / (np.sum(exps, axis=1, keepdims=True) + 1e-9)

    def apply_activation_derivative(self, a, activation):
        if activation == "sigmoid":
            return a * (1 - a)
        elif activation == "relu":
            return np.where(a > 0, 1, 0)
        elif activation == "linear":
            return 1
        elif activation == "cube":
            return 2 * a ** 2


    def forward(self, X):
        self.activations_values = [X]
        self.z_values = []

        for i in range(self.num_layers - 1):
            z = np.dot(self.activations_values[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.apply_activation(z, self.activations[i])
            self.activations_values.append(a)
        return self.activations_values[-1]

    def backward(self, X, y, output, learning_rate):
        deltas = []

        if self.activations[-1] == "softmax" and self.loss_function == "cross_entropy":
            output_delta =  y - output
        else:
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


    def train(self, X, y, epochs, learning_rate, print_loss=True):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            
            if epoch % 1000 == 0 and print_loss:
                if self.loss_function == "cross_entropy":
                    loss = cross_entropy_loss(y, output)
                else:
                    loss = mean_squared_error(y, output)
                print(f'Epoch {epoch}, Loss: {loss}')

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

    def predict(self, X, regression = False):
        predictions = self.forward(X)
        if(regression):
            return predictions
        else:    
            predictions = self.forward(X)
            return np.argmax(predictions, axis=1)


    
    
def calculate_accuracy(y_true, y_pred):
    y_true_labels = np.argmax(y_true, axis=1)
    #print(y_true_labels)
    #print(y_pred)
    correct_predictions = np.sum(y_true_labels == y_pred)
    accuracy = (correct_predictions / len(y_true)) * 100
    return accuracy



def perform_tests_simple(path):
    #Accuracy for data.simple.test.100:  99.0 %
    #Accuracy for data.simple.test.500:  99.4 %
    #Accuracy for data.simple.test.1000:  99.6 %
    #Accuracy for data.simple.test.10000:  99.64 %
    layers = [2, 4, 2]
    activations = ["linear","softmax"]  #"relu", "sigmoid" or "linear","softmax"
    learning_rate = 0.0001
    epochs = 10000
    seed = 42
    loss_function = "cross_entropy"  #"cross_entropy" or "mse"

    numbers = [100, 500, 1000, 10000]
    for num in numbers:
        train_file_path = path + f"data.simple.train.{num}.csv"

        data = pd.read_csv(train_file_path, delimiter=',', header=0)
        data = data.sample(frac=1).reset_index(drop=True)
        X = data[['x', 'y']].to_numpy()
        cls = data['cls'].to_numpy() - 1 
        y = np.eye(2)[cls]

        nn = Network(layers, activations, loss_function=loss_function, seed=seed)
        nn.train(X, y, epochs, learning_rate, False)

        test_file_path = path + f"data.simple.test.{num}.csv"

        data = pd.read_csv(test_file_path, delimiter=',', header=0)
        X = data[['x', 'y']].to_numpy()
        cls = data['cls'].to_numpy() - 1 
        y = np.eye(2)[cls]
        predictions = nn.predict(X)
    
        print(f"Accuracy for data.simple.test.{num}: ", calculate_accuracy(y, predictions), "%")

    return

def perform_tests_three_gauss(path):
    layers = [2, 4, 3]
    activations = ["linear", "softmax"]
    learning_rate = 0.0001
    epochs = 10000
    seed = 41
    loss_function = "cross_entropy"

    numbers = [100, 500, 1000, 10000]
    for num in numbers:
        train_file_path = path + f"data.three_gauss.train.{num}.csv"

        data = pd.read_csv(train_file_path, delimiter=',', header=0)
        data = data.sample(frac=1).reset_index(drop=True)
        X = data[['x', 'y']].to_numpy()
        cls = data['cls'].to_numpy() - 1
        y = np.eye(3)[cls]

        nn = Network(layers, activations, loss_function=loss_function, seed=seed)
        nn.train(X, y, epochs, learning_rate, False)

        test_file_path = path + f"data.three_gauss.test.{num}.csv"
        data = pd.read_csv(test_file_path, delimiter=',', header=0)
        X = data[['x', 'y']].to_numpy()
        cls = data['cls'].to_numpy() - 1 
        y = np.eye(3)[cls]
        
        predictions = nn.predict(X)
        print(f"Accuracy for data.three_gauss.test.{num}: ", calculate_accuracy(y, predictions), "%")
        nn.plot_error_history()
    return

def perform_tests_activation(path):
    #Mean Squared Error for data.regression.test.100: 0.06964267611614208
    #Mean Squared Error for data.regression.test.500: 0.1099453715478068
    #Mean Squared Error for data.regression.test.1000: 0.011450350054929089
    #Mean Squared Error for data.regression.test.10000: 0.1541758763697975
    layers = [1,4, 1]  
    activations = ["sigmoid","linear"] 
    #learning_rate = 0.01
    epochs = 10000
    seed = 42
    loss_function = "mse" 

    numbers = [100, 500, 1000, 10000]  
    for num in numbers:
        learning_rate = 1.0/num
        train_file_path = path + f"data.activation.train.{num}.csv"
        data = pd.read_csv(train_file_path, delimiter=',', header=0)
        data = data.sample(frac=1).reset_index(drop=True)  

        X = data[['x']].to_numpy() 
        y = data[['y']].to_numpy() 

        #standarization of data
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        X_standardized = (X - mean_X) / std_X
        
        mean_y = np.mean(y, axis=0)
        std_y = np.std(y, axis=0)
        y_standardized = (y - mean_y) / std_y

        nn = Network(layers, activations, loss_function=loss_function, seed=seed)
        nn.train(X_standardized, y_standardized, epochs, learning_rate, False)  

        test_file_path = path + f"data.activation.test.{num}.csv"
        data = pd.read_csv(test_file_path, delimiter=',', header=0)
        X_test = data[['x']].to_numpy()  
        y_test = data[['y']].to_numpy() 
        X_test_standardized = (X_test - mean_X) / std_X


        predictions_standardized = nn.predict(X_test_standardized, True)
        predictions = predictions_standardized * std_y + mean_y

        mse = mean_squared_error(y_test, predictions)
        print(f"Mean Squared Error for data.regression.test.{num}: {mse}")

    return

def perform_tests_cube(path):
    #Mean Squared Error for data.regression.test.100: 51015.399142573246
    #Mean Squared Error for data.regression.test.500: 50003.65932079104
    #Mean Squared Error for data.regression.test.1000: 52768.59193611867
    #Mean Squared Error for data.regression.test.10000: 50443.42037943883
    layers = [1,1,1]  
    activations = ["cube","linear"] 
    #learning_rate = 0.01
    epochs = 10000
    seed = 42
    loss_function = "mse" 

    numbers = [100, 500, 1000, 10000]  
    for num in numbers:
        learning_rate = 1.0/num
        train_file_path = path + f"data.cube.train.{num}.csv"
        data = pd.read_csv(train_file_path, delimiter=',', header=0)
        data = data.sample(frac=1).reset_index(drop=True)

        X = data[['x']].to_numpy() 
        y = data[['y']].to_numpy() 

        #standarization of data
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        X_standardized = (X - mean_X) / std_X
        
        mean_y = np.mean(y, axis=0)
        std_y = np.std(y, axis=0)
        y_standardized = (y - mean_y) / std_y

        nn = Network(layers, activations, loss_function=loss_function, seed=seed)
        nn.train(X_standardized, y_standardized, epochs, learning_rate, False)  

        test_file_path = path + f"data.cube.test.{num}.csv"
        data = pd.read_csv(test_file_path, delimiter=',', header=0)
        X_test = data[['x']].to_numpy()  
        y_test = data[['y']].to_numpy() 
        X_test_standardized = (X_test - mean_X) / std_X


        predictions_standardized = nn.predict(X_test_standardized, True)
        predictions = predictions_standardized * std_y + mean_y

        mse = mean_squared_error(y_test, predictions)
        print(f"Mean Squared Error for data.regression.test.{num}: {mse}")

    return

if __name__ == "__main__":
    
#Accuracy for data.simple.test.100:  99.0 %
    #Accuracy for data.simple.test.500:  99.4 %
    #Accuracy for data.simple.test.1000:  99.6 %
    #Accuracy for data.simple.test.10000:  99.64 %
    # path = 'S:/SN/projekt1/classification/'
    # train_file_path = path + f"data.activation.train.{100}.csv"
    # layers = [2, 4, 2]
    # activations = ["linear","sigmoid"]  #"relu", "sigmoid" or "linear"
    # learning_rate = 0.001
    # epochs = 10000
    # seed = 42
    # loss_function = "cross_entropy"  #"cross_entropy" or "mse"

    # numbers = [100]
    # for num in numbers:

    #     data = pd.read_csv(train_file_path, delimiter=',', header=0)
    #     data = data.sample(frac=1).reset_index(drop=True)
    #     X = data[['x', 'y']].to_numpy()
    #     cls = data['cls'].to_numpy() - 1 
    #     y = np.eye(2)[cls]

    #     nn = Network(layers, activations, loss_function=loss_function, seed=seed)
    #     nn.train(X, y, epochs, learning_rate, False)

    #     test_file_path = path + f"data.simple.test.{num}.csv"

    #     data = pd.read_csv(test_file_path, delimiter=',', header=0)
    #     X = data[['x', 'y']].to_numpy()
    #     cls = data['cls'].to_numpy() - 1 
    #     y = np.eye(2)[cls]
    #     predictions = nn.predict(X)
    
    #     print(f"Accuracy for data.simple.test.{num}: ", calculate_accuracy(y, predictions), "%")

    # nn.plot_error_history()

#    print("Accuracy: ", calculate_accuracy(y, predictions), "%")

    folder_path = 'S:/SN/projekt1/classification/'
    perform_tests_simple(folder_path)
    #perform_tests_three_gauss(folder_path)
    folder_path = 'S:/SN/projekt1/regression/'
    #perform_tests_activation(folder_path)
    #perform_tests_cube(folder_path)



    # train_file_path = folder_path + f"data.activation.train.{100}.csv"
    # data = pd.read_csv(train_file_path, delimiter=',', header=0)
    # data = data.sample(frac=1).reset_index(drop=True)  
    # X = data['x'] 
    # y = data['y'] 
    # matplotlib.pyplot.plot(X,y)
    # matplotlib.pyplot.show()