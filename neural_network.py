import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def sigmoid(z):
    z = np.clip(z, -500, 500)  # avoid overflow
    return 1 / (1 + np.exp(-z))
def derivative_sigmoid(x): return x * (1 - x)
def cost_function(Y_predicted, Y_expected): return 1 / np.shape(Y_expected)[0] * np.sum((Y_predicted - Y_expected) ** 2)

def generate_parameters(variables_number: int, hidden_neurons: int, output_neurons: int) -> dict:
    parameters = {}
    hidden_weights = np.random.uniform(low=-0.2, high=0.2, size=(variables_number, hidden_neurons))
    output_weights = np.random.uniform(low=-2, high=2, size=(hidden_neurons, output_neurons))
    parameters['weights'] = [hidden_weights, output_weights]
    parameters['biases'] = [np.ones([1, hidden_neurons]), np.ones([1, output_neurons])]
    return parameters

def forward_propagation(input_layer: np.array, parameters: dict):
    hl = np.dot(input_layer, parameters['weights'][0]) + parameters['biases'][0]
    hidden_layer = sigmoid(hl)
    ol = np.dot(hidden_layer, parameters['weights'][1]) + parameters['biases'][1]
    output_layer = sigmoid(ol)
    return [input_layer, hidden_layer, output_layer]

def backward_propagation(layers: list, cost: float, parameters: dict):
    learning_rate = 0.1

    for layer_idx in range(2, 0, -1):
        error_delta = cost * derivative_sigmoid(layers[layer_idx])
        parameters['weights'][layer_idx - 1] += learning_rate * np.dot(np.transpose(layers[layer_idx - 1]), error_delta)
        parameters['biases'][layer_idx - 1] += learning_rate * np.sum(error_delta, axis=0)
        cost = np.dot(error_delta, np.transpose(parameters['weights'][layer_idx - 1]))
    return parameters

def train_neural_network(X: np.array, Y: np.array, epochs: int, hidden_neurons: int, output_neurons: int):
    parameters = generate_parameters(np.shape(X)[1], hidden_neurons, output_neurons)
    for epoch in range(epochs):
        layers = forward_propagation(X, parameters)
        #print(f"Epoch: {epoch}\nHidden layer:\n{layers[1]}\nOutput layer:\n{layers[2]}\n")
        output_layer = layers[2]
        cost = cost_function(np.squeeze(output_layer), Y)
        parameters = backward_propagation(layers, cost, parameters)
        #print(parameters)
    return parameters


def calculate_accuracy(x, y, parameters):
    total = y.shape[0]
    count = 0
    for x, y in zip(x, y):
        output = forward_propagation(x, parameters)[-1]
        if output.argmax() == y.argmax():
                count += 1
        return count/total

def evaluate():
    # Import data
    file = r'winequality-white.csv'
    my_data = np.genfromtxt(file, delimiter=';')
    my_data = my_data[2:,:] # drop headers
    x = my_data[:,:-1]
    y = my_data[:,-1]
    #x = x / np.linalg.norm(x)
    y = pd.get_dummies(y).to_numpy()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)
    parameters = train_neural_network(x_train, y_train, 100, 21, 7)
    # validate
    accuracy = calculate_accuracy(x_val, y_val, parameters)
    print(f"Accuracy: {accuracy * 100}")

print(evaluate())
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural network implementation.")
    parser.add_argument("-f", "--file", required=True, help="path to the file containing dataset")
    parser.add_argument("-i", "--input", type=int, help="number of input neurons", required=True)
    parser.add_argument("-hd", "--hidden", type=int, help="number of hidden neurons", required=True)
    parser.add_argument("-o", "--output", type=int, help="number of output neurons", required=True)
    args = vars(parser.parse_args())
'''