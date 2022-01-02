import numpy as np

def sigmoid(x, Beta=1): return 1 / (1 + np.exp(-Beta * x))
def derivative_sigmoid(x): return x * (1 - x)
def cost_function(Y_predicted, Y_expected): return 1 / np.shape(Y_expected)[0] * np.sum((Y_predicted - Y_expected) ** 2)
    
def generate_parameters(variables_number: int, hidden_neurons: int, output_neurons: int) -> dict:
    parameters = {}
    parameters['weights'] = [np.ones([hidden_neurons, variables_number]), np.ones([output_neurons, hidden_neurons])]
    parameters['bias'] = [np.ones([hidden_neurons, 1]), np.ones([output_neurons, 1])]
    return parameters

def forward_propagation(input_layer: np.array, parameters: dict, hidden_neurons: int, output_neurons: int):
    h = np.dot(input_layer, parameters['weights'][0])
    hidden_layer = sigmoid(h)

    o = np.dot(hidden_layer, parameters['weights'][1])
    output_layer = sigmoid(o)

    return [input_layer, hidden_layer, output_layer]

def backward_propagation(layers: list, cost: float, parameters: dict):
    weights = parameters['weights']
    learning_rate = 0.1

    for layer_idx in range(2, 0, -1):
        error_delta = cost * derivative_sigmoid(layers[layer_idx])
        weights[layer_idx - 1] += learning_rate * np.dot(np.transpose(layers[layer_idx - 1]), error_delta)
        cost = np.dot(error_delta, np.transpose(weights[layer_idx - 1]))
    return parameters

def train_neural_network(X: np.array, Y: np.array, epochs: int, hidden_neurons: int, output_neurons: int):
    parameters = generate_parameters(np.shape(X)[1], hidden_neurons, output_neurons)
    for epoch in range(epochs):
        layers = forward_propagation(X, parameters, hidden_neurons, output_neurons)
        output_layer = layers[2]
        cost = cost_function(np.squeeze(output_layer), Y)
        parameters = backward_propagation(layers, cost, parameters)
    return parameters

X = np.array((
    [7, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3, 0.45, 8.8], 
    [6.3, 0.3, 0.34, 1.6, 0.049, 14,132, 0.994, 3.3, 0.49, 9.5], 
    [8.1, 0.28, 0.4, 6.9, 0.05, 30,97, 0.9951, 3.26, 0.44, 10.1],
    [7.2, 0.23, 0.32, 8.5, 0.058, 47, 186, 0.9956, 3.19, 0.4, 9.9],
    [7.2, 0.23, 0.32, 8.5, 0.058, 47, 186, 0.9956, 3.19, 0.4, 9.9],
    [8.1, 0.28, 0.4, 6.9, 0.05, 30, 97, 0.9951, 3.26, 0.44, 10.1],
    [6.2, 0.32, 0.16, 7, 0.045, 30, 136, 0.9949, 3.18, 0.47, 9.6]
), dtype=float)

y = np.array(([6], [6], [6], [6], [6], [6], [6]), dtype=float)
col_count = np.shape(X)[1]

a = train_neural_network(X, y, 10, col_count, col_count)
print(a)
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural network implementation.")
    parser.add_argument("-f", "--file", required=True, help="path to the file containing dataset")
    parser.add_argument("-i", "--input", type=int, help="number of input neurons", required=True)
    parser.add_argument("-hd", "--hidden", type=int, help="number of hidden neurons", required=True)
    parser.add_argument("-o", "--output", type=int, help="number of output neurons", required=True)
    args = vars(parser.parse_args())
'''