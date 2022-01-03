import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
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
        cost = np.dot(error_delta, np.transpose(parameters['weights'][layer_idx - 1]))
    return parameters

def train_neural_network(X: np.array, Y: np.array, epochs: int, hidden_neurons: int, output_neurons: int):
    parameters = generate_parameters(np.shape(X)[1], hidden_neurons, output_neurons)
    for epoch in range(epochs):
        layers = forward_propagation(X, parameters)
        output_layer = layers[2]
        cost = cost_function(np.squeeze(output_layer), Y)
        parameters = backward_propagation(layers, cost, parameters)
        print(parameters['weights'])
        print()
    return parameters

'''
X = [
    [7, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3, 0.45, 8.8], 
    [6.3, 0.3, 0.34, 1.6, 0.049, 14,132, 0.994, 3.3, 0.49, 9.5], 
    [8.1, 0.28, 0.4, 6.9, 0.05, 30,97, 0.9951, 3.26, 0.44, 10.1],
    [7.2, 0.23, 0.32, 8.5, 0.058, 47, 186, 0.9956, 3.19, 0.4, 9.9],
    [7.2, 0.23, 0.32, 8.5, 0.058, 47, 186, 0.9956, 3.19, 0.4, 9.9],
    [8.1, 0.28, 0.4, 6.9, 0.05, 30, 97, 0.9951, 3.26, 0.44, 10.1],
    [6.2, 0.32, 0.16, 7, 0.045, 30, 136, 0.9949, 3.18, 0.47, 9.6]
]

y = np.array(([6], [6], [6], [6], [6], [6], [6]), dtype=float)
'''

def evaluate():
    file = open("winequality-white.csv")
    data = np.loadtxt(file, delimiter=';', skiprows=1)
    test_X = data[:1959,:-1]
    test_y = data[:1959,-1]
    train_X = data[1959:,:-1]
    train_y = data[1959:,-1]
    col_count = np.shape(train_X)[1]
    test_y = np.expand_dims(test_y, -1)
    train_y = np.expand_dims(train_y, -1)
    nn_trained = train_neural_network(train_X, train_y, 10, col_count, col_count)
    print(nn_trained)
    # walidacja
    class_val = forward_propagation(test_X, nn_trained)[2]
    print()
    print(class_val[2])
    return 1

evaluate()
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural network implementation.")
    parser.add_argument("-f", "--file", required=True, help="path to the file containing dataset")
    parser.add_argument("-i", "--input", type=int, help="number of input neurons", required=True)
    parser.add_argument("-hd", "--hidden", type=int, help="number of hidden neurons", required=True)
    parser.add_argument("-o", "--output", type=int, help="number of output neurons", required=True)
    args = vars(parser.parse_args())
'''