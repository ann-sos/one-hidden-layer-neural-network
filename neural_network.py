import numpy as np
import pandas as pd
import argparse




def sumator(inputs, weights, bias):
    return (weights*inputs).sum(axis=1) + bias

def sigmoid(x, Beta=1):
    "Activation function"
    return 1/(1 + np.exp(- Beta * x))

def derivative_sigmoid(x):
    return x * (1 - x)

def cost_function(Y_predicted: pd.DataFrame, Y_expected: pd.DataFrame):
    """MSE= Mean Squared error"""
    return 1 / Y_expected.shape[0] * np.sum((Y_predicted - Y_expected) ** 2)
    
def proportion_split_dataset(dataset: pd.DataFrame, proportion: float):
    # split dataset into training and testing set
    training_df = dataset.sample(frac=proportion)
    testing_df = dataset.drop(training_df.index)
    return training_df, testing_df

def generate_parameters(variables_number: int, hidden_neurons: int, output_neurons: int) -> dict:
    def initialise(x, y):
        # returns numpy array of ones
        if x > 1: return np.ones([x, y])
        else: return 1
    parameters = {}
    parameters['weights'] = [initialise(hidden_neurons, variables_number), initialise(output_neurons, hidden_neurons)] #input layer and hidden layer
    parameters['bias'] = [initialise(hidden_neurons, 1), initialise(output_neurons, 1)]
    return parameters

def forward_propagation(
    input_layer: pd.DataFrame, 
    hidden_neurons: int, 
    output_neurons: int, 
    parameters: dict
):
    hidden_layer = pd.DataFrame()
    output_layer = pd.DataFrame()

    def ifarray(x, i):
        if not isinstance(x, np.ndarray): return x
        else: return x[i]
    def ifarray2(x, i):
        if not isinstance(x, np.ndarray): return x
        else: return x[i,:]
    # initialize hidden layer dataframe
    for i in range(hidden_neurons):
        hidden_layer = hidden_layer.assign(**{f"A{i+1}": np.ones(input_layer.shape[0])})
    for i in range(hidden_neurons):
        hidden_layer[f"A{i+1}"] = sigmoid(sumator(input_layer, ifarray2(parameters['weights'][0], i), ifarray(parameters['bias'][0], i))) 
    # initialize output layer
    for i in range(output_neurons):
        output_layer = output_layer.assign(**{f"B{i+1}": np.ones(input_layer.shape[0])})
    for i in range(output_neurons):
        output_layer[f"B{i+1}"] = sigmoid(sumator(hidden_layer, ifarray2(parameters['weights'][1], i), ifarray(parameters['bias'][1], i))) 
    
    return [
        input_layer, 
        hidden_layer,
        output_layer, 
    ]

def backward_propagation(layers: list, cost: float, parameters: dict):
    weights = parameters['weights']

    for layer_idx in range(2, 0, -1):
        error_delta = cost * derivative_sigmoid(layers[layer_idx])
        #update_weights here
        cost = np.dot(error_delta, np.transpose(weights[layer_idx - 1]))

    return parameters


def train_neural_network(X: pd.DataFrame, Y: pd.DataFrame, epochs: int, hidden_neurons: int, output_neurons: int):
    parameters = generate_parameters(X.shape[1], hidden_neurons, output_neurons)
    for epoch in range(epochs):
        layers = forward_propagation(X, hidden_neurons, output_neurons, parameters)
        output_layer = layers[2]
        cost = cost_function(output_layer.squeeze(), Y)
        parameters = backward_propagation(layers, cost, parameters)
    return parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural network implementation.")
    parser.add_argument("-f", "--file", required=True, help="path to the file containing dataset")
    parser.add_argument("-i", "--input", type=int, help="number of input neurons", required=True)
    parser.add_argument("-hd", "--hidden", type=int, help="number of hidden neurons", required=True)
    parser.add_argument("-o", "--output", type=int, help="number of output neurons", required=True)
    args = vars(parser.parse_args())
