import numpy as np

def sigmoid(x, Beta=1): return 1 / (1 + np.exp(-Beta * x))
def derivative_sigmoid(x): return x * (1 - x)
def cost_function(Y_predicted, Y_expected): return 1 / np.shape(Y_expected)[0] * np.sum((Y_predicted - Y_expected) ** 2)
    
def generate_parameters(variables_number: int, hidden_neurons: int, output_neurons: int) -> dict:
    parameters = {}
    parameters['weights'] = [np.ones([variables_number, hidden_neurons]), np.ones([hidden_neurons, output_neurons])]
    parameters['bias'] = [np.ones([1, hidden_neurons]), np.ones([1, output_neurons])]
    return parameters

def forward_propagation(input_layer: np.array, parameters: dict):
    hidden_layer = sigmoid(np.dot(input_layer, parameters['weights'][0]) + parameters['bias'][0])
    output_layer = sigmoid(np.dot(hidden_layer, parameters['weights'][1]) + parameters['bias'][1])
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
        layers = forward_propagation(X, parameters)
        output_layer = layers[2]
        cost = cost_function(np.squeeze(output_layer), Y)
        parameters = backward_propagation(layers, cost, parameters)
    return parameters

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural network implementation.")
    parser.add_argument("-f", "--file", required=True, help="path to the file containing dataset")
    parser.add_argument("-i", "--input", type=int, help="number of input neurons", required=True)
    parser.add_argument("-hd", "--hidden", type=int, help="number of hidden neurons", required=True)
    parser.add_argument("-o", "--output", type=int, help="number of output neurons", required=True)
    args = vars(parser.parse_args())
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
    nn_trained = train_neural_network (train_X, train_y, 10, col_count, col_count)
    # walidacja
    class_val = forward_propagation(test_X, nn_trained)[2]
    print(class_val)
    return 1

evaluate()