import numpy as np
import cupy as cp
from sklearn.metrics import roc_auc_score

class WeatherPredictionNetwork:
    def __init__(self, layers, activations, seed=None, l2_lambda=0.01):
        if seed is not None:
            cp.random.seed(seed)

        self.num_layers = len(layers)
        self.weights = []
        self.biases = []
        self.activations = activations
        self.l2_lambda = l2_lambda

        for i in range(self.num_layers - 1):
            limit = cp.sqrt(6 / (layers[i] + layers[i + 1]))
            self.weights.append(cp.random.uniform(-limit, limit, (layers[i], layers[i + 1])).astype(cp.float32))
            self.biases.append(cp.zeros((1, layers[i + 1]), dtype=cp.float32))

    def apply_activation(self, z, activation):
        if activation == "sigmoid":
            return 1 / (1 + cp.exp(-cp.clip(z, -10, 10)))
        elif activation == "relu":
            return cp.maximum(0, z)
        elif activation == "tanh":
            return cp.tanh(z)
        elif activation == "linear":
            return z
        elif activation == "softmax":
            z_stable = z - cp.max(z, axis=1, keepdims=True)
            exps = cp.exp(z_stable)
            return exps / cp.sum(exps, axis=1, keepdims=True)

    def apply_activation_derivative(self, a, activation):
        if activation == "sigmoid":
            return a * (1 - a)
        elif activation == "relu":
            return cp.where(a > 0, 1, 0)
        elif activation == "tanh":
            return 1 - a**2
        elif activation == "linear":
            return 1

    def forward(self, X):
        self.activations_values = [X]
        self.z_values = []

        for i in range(self.num_layers - 2):
            z = cp.dot(self.activations_values[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.apply_activation(z, self.activations[i])
            self.activations_values.append(a)

        z_last = cp.dot(self.activations_values[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_last)
        
        z_reg, z_cls = z_last[:, 0:1], z_last[:, 1:]
        a_reg = self.apply_activation(z_reg, "linear")
        a_cls = self.apply_activation(z_cls, "sigmoid")
        
        self.activations_values.append(cp.hstack([a_reg, a_cls]))

        return self.activations_values[-1]

    def backward(self, X, y, output, learning_rate):
        deltas = []

        y_reg, y_cls = y[:, 0:1], y[:, 1:]
        output_reg, output_cls = output[:, 0:1], cp.clip(output[:, 1:], 1e-9, 1 - 1e-9)

        reg_error = y_reg - output_reg
        reg_delta = reg_error * self.apply_activation_derivative(output_reg, "linear")

        cls_error = y_cls - output_cls
        cls_delta = cls_error * self.apply_activation_derivative(output_cls, "sigmoid")

        output_delta = cp.hstack([reg_delta, cls_delta])
        deltas.append(output_delta)

        for i in range(self.num_layers - 2, 0, -1):
            z = self.z_values[i - 1]
            delta = cp.dot(deltas[-1], self.weights[i].T) * self.apply_activation_derivative(self.activations_values[i], self.activations[i - 1])
            deltas.append(delta)

        deltas.reverse()

        for i in range(self.num_layers - 1):
            grad_w = cp.dot(self.activations_values[i].T, deltas[i]) + self.l2_lambda * self.weights[i]  # Add L2 regularization
            grad_b = cp.sum(deltas[i], axis=0, keepdims=True)

            grad_w = cp.clip(grad_w, -1.0, 1.0)
            grad_b = cp.clip(grad_b, -1.0, 1.0)

            self.weights[i] += learning_rate * grad_w
            self.biases[i] += learning_rate * grad_b

    def clip_weights(self, clip_value=1.0):
        for i in range(len(self.weights)):
            self.weights[i] = cp.clip(self.weights[i], -clip_value, clip_value)

    def train(self, X, y, epochs, learning_rate, lower_rate = [500]):

        for epoch in range(epochs):
            if epoch in lower_rate:
                learning_rate = learning_rate / 10
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            #self.clip_weights()

            if epoch % 100 == 0:
                reg_loss = cp.mean(cp.abs(y[:, 0] - output[:, 0]))
                #reg_loss2 = cp.mean(cp.abs(y[:, 1] - output[:, 1]))
                auc = roc_auc_score(cp.asnumpy(y[:, 1]), cp.asnumpy(output[:, 1]))
                print(f"Epoch {epoch}, Regression Loss: {reg_loss}, Classification AUC: {auc}, Learning Rate: {learning_rate}")

    def predict(self, X):
        output = self.forward(X)
        reg_output = output[:, 0]
        cls_output = (output[:, 1]).astype(cp.float32)
        return cp.hstack([reg_output[:, None], cls_output[:, None]])
