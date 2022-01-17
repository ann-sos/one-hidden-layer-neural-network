import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def sigmoid(x):
    x = np.clip(x, -500, 500)  # avoid overflow
    return 1 / (1 + np.exp(-x))
def derivative_sigmoid(x): 
    x = sigmoid(x)
    return x * (1 - x)

def generate_parameters(variables_number: int, hidden_neurons: int, output_neurons: int) -> dict:
    parameters = {}
    np.random.seed(42)
    hidden_weights = np.random.uniform(low=-0.2, high=0.2, size=(hidden_neurons, variables_number,))
    output_weights = np.random.uniform(low=-2, high=2, size=(output_neurons, hidden_neurons))
    parameters['weights'] = [hidden_weights, output_weights]
    parameters['biases'] = [np.random.randn(1, hidden_neurons).squeeze(), np.random.randn(1, output_neurons).squeeze()]
    return parameters

def forward_propagation(input_layer: np.array, parameters: dict):
    hl = np.dot(parameters['weights'][0], input_layer) + parameters['biases'][0]
    hidden_layer = sigmoid(hl)
    ol = np.dot(parameters['weights'][1], hidden_layer) + parameters['biases'][1]
    output_layer = sigmoid(ol)
    return [input_layer, hidden_layer, output_layer, hl, ol]

def backward_propagation(layers: list, Y, parameters: dict):
    learning_rate = 0.01
    change_w = {}

    # Calculate W2 update
    error = 2 * (layers[2] - Y) / layers[2].shape[0] * derivative_sigmoid(layers[4])
    change_w['W2'] = np.outer(error, layers[1])
    change_w['B2'] = error


    # Calculate W1 update
    error = np.dot(parameters['weights'][1].T, error) * derivative_sigmoid(layers[3])
    change_w['W1'] = np.outer(error, layers[0])
    change_w['B1'] = error
    #print("change_w['B2']", change_w['B2'])
    #Update parameters
    parameters['weights'][0] -= learning_rate * change_w['W1']
    parameters['biases'][0] -= learning_rate * change_w['B1']
    parameters['weights'][1] -= learning_rate * change_w['W2']
    parameters['biases'][1] -= learning_rate * change_w['B2']
    
    """for layer_idx in range(2, 0, -1):
        error_delta = cost * derivative_sigmoid(layers[layer_idx])
        #print(f"Shape of error delta: {np.shape(error_delta)}\n")
        #print(f"Shape of bias: {np.shape(parameters['biases'][layer_idx - 1])}\n")
        parameters['weights'][layer_idx - 1] += learning_rate * np.dot(np.transpose(layers[layer_idx - 1]), error_delta)
        parameters['biases'][layer_idx - 1] += learning_rate * np.sum(error_delta, axis=0)
        cost = np.dot(error_delta, np.transpose(parameters['weights'][layer_idx - 1]))"""
    return parameters

def train_neural_network(X: np.array, Y: np.array, epochs: int, hidden_neurons: int, output_neurons: int):
    parameters = generate_parameters(np.shape(X)[1], hidden_neurons, output_neurons)
    for epoch in range(epochs):
        for x, y in zip(X, Y):
            layers = forward_propagation(x, parameters)
            parameters = backward_propagation(layers, y, parameters)
    #print("parameters", parameters)
    #print(f"Epoch: {epoch}\nOutput layer:\n{layers[2]}\n")
    return parameters


def calculate_accuracy(X, Y, parameters):
    total = Y.shape[0]
    count = 0
    for x, y in zip(X, Y):
        output = forward_propagation(x, parameters)[-1]
        #print(f"prediction: {output} \n expected: {y}\n")

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
    x = x / np.linalg.norm(x)
    y = pd.get_dummies(y).to_numpy()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)
    parameters = train_neural_network(x_train, y_train, 500, 21, 7)
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