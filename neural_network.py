import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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

def backward_propagation(layers: list, Y, parameters: dict, learning_rate=0.1):
    #learning_rate = 0.01
    change_w = {}

    # Calculate W2 update
    error = 2 * (layers[2] - Y) / layers[2].shape[0] * derivative_sigmoid(layers[4])
    change_w['W2'] = np.outer(error, layers[1])
    change_w['B2'] = error


    # Calculate W1 update
    error = np.dot(parameters['weights'][1].T, error) * derivative_sigmoid(layers[3])
    change_w['W1'] = np.outer(error, layers[0])
    change_w['B1'] = error
    #Update parameters
    parameters['weights'][0] -= learning_rate * change_w['W1']
    parameters['biases'][0] -= learning_rate * change_w['B1']
    parameters['weights'][1] -= learning_rate * change_w['W2']
    parameters['biases'][1] -= learning_rate * change_w['B2']
    
    return parameters

def train_neural_network(X: np.array, Y: np.array, X_test: np.array, Y_test: np.array, epochs: int, hidden_neurons: int, output_neurons: int, learning_rate=0.1, test_on_the_go=False):
    accuracy_list = []
    parameters = generate_parameters(np.shape(X)[1], hidden_neurons, output_neurons)
    for epoch in range(epochs):
        for x, y in zip(X, Y):
            layers = forward_propagation(x, parameters)
            parameters = backward_propagation(layers, y, parameters, learning_rate)
        if epoch % 10 == 0 and test_on_the_go:
            accuracy = calculate_accuracy(X_test, Y_test, parameters)
            accuracy_list.append(accuracy)
            print(f"Epoch: {epoch} --- Accuracy: {accuracy:.2f}%")

    return parameters, accuracy_list


def calculate_accuracy(X, Y, parameters):
    total = Y.shape[0]
    count = 0
    for x, y in zip(X, Y):
        output = forward_propagation(x, parameters)[-1]
        if output.argmax() == y.argmax():
                count += 1
    return count/total * 100

def import_data(file):
    # Import data
    my_data = np.genfromtxt(file, delimiter=';')
    my_data = my_data[2:,:] # drop headers
    x = my_data[:,:-1]
    y = my_data[:,-1]
    #x = x / np.linalg.norm(x)
    y = pd.get_dummies(y).to_numpy()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)
    return x_train, x_val, y_train, y_val


def evaluate(file, learning_rate=0.01):
    x_train, x_val, y_train, y_val = import_data(file)
    parameters, accuracy_lst = train_neural_network(x_train, y_train, x_val, y_val, 100, 21, 7)
    # validate
    accuracy = calculate_accuracy(x_val, y_val, parameters)
    print(f"Accuracy: {accuracy * 100:.2f}")
    return accuracy_lst

#Test learning rate influence on accuracy
x_train, x_val, y_train, y_val = import_data(r'winequality-white.csv')

plt.figure(1)
for learning_rate in [0.001, 0.01, 0.5]:
    parameters, accuracy_lst = train_neural_network(x_train, y_train, x_val, y_val, 200, 21, 7, learning_rate, test_on_the_go=True)
    plt.plot(np.arange(0, len(accuracy_lst)*10, 10), accuracy_lst, label=f"learning_rate={learning_rate}")
plt.title("Neural network")
plt.xlabel("Epoch [-]")
plt.ylabel("Accuracy [%]")
plt.legend()
plt.savefig("Neural_network_accuracy_epoch_learning_rate")

plt.figure(2)
for hidden_neurons in [14, 28, 56]:
    parameters, accuracy_lst = train_neural_network(x_train, y_train, x_val, y_val, 200, hidden_neurons, 7, test_on_the_go=True)
    plt.plot(np.arange(0, len(accuracy_lst)*10, 10), accuracy_lst, label=f"hidden_neurons={hidden_neurons}")
plt.title("Neural network")
plt.xlabel("Epoch [-]")
plt.ylabel("Accuracy [%]")
plt.legend()
plt.savefig("Neural_network_accuracy_epoch_hidden")


plt.figure(3)
parameters, accuracy_lst = train_neural_network(x_train, y_train, x_val, y_val, 500, 21, 7, learning_rate=0.05, test_on_the_go=True)
plt.plot(np.arange(0, len(accuracy_lst)*10, 10), accuracy_lst)
plt.title("Neural network")
plt.xlabel("Epoch [-]")
plt.ylabel("Accuracy [%]")
plt.savefig("Neural_network_accuracy_epoch_long")