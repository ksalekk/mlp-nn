import numpy as np

import csv
from network_building import NeuralNetwork
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def load_data(filename, delim=',', encode="utf-8"):
    with open(filename, "rt", newline="", encoding=encode) as file:
        reader = csv.reader(file, delimiter=delim)
        list_data = list(reader)
    return list_data



def main_test():
    # load training and testing dataset
    training_np = np.asarray(load_data(filename="../data/training_data.csv"))[1:, 2:]
    testing_np = np.asarray(load_data(filename="../data/testing_data.csv"))[1:, 2:]

    # scaling properties values to (0, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_training = np.hstack( (scaler.fit_transform(training_np[:, :4]), training_np[:, 4:5]) ).astype(float)
    scaled_testing = np.hstack( (scaler.fit_transform(testing_np)[:, :4], testing_np[:, 4:5]) ).astype(float)


    learning_rate = 0.4
    mlp_network = NeuralNetwork(4, [3], 1, learning_rate)


    err_function = mlp_network.learn_network(scaled_training, iterations_cnt=1000, sequentially=False)
    plot_error_function(err_function, learning_rate=f'{learning_rate}', step=1, ymin=100, ymax=400)

    predicted_vals = mlp_network.predict(scaled_testing[:, 0:4])
    true_vals = scaled_testing[:, 4].astype(int)

    threshold = np.median(predicted_vals)
    formatted_predicted_vals = (predicted_vals >= threshold).astype(int).squeeze()

    tn, fp, fn, tp = confusion_matrix(true_vals, formatted_predicted_vals).ravel()
    accuracy = (tn + tp) / len(formatted_predicted_vals)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(f"Accuracy = {accuracy}")
    print(f"Sensitivity = {sensitivity}")
    print(f"Specificity = {specificity}")



def plot_error_function(error_function, learning_rate='?', step=1, ymin=0, ymax=100):
    plt.figure()

    x = np.arange(0, len(error_function), step)
    y = error_function[x]
    plt.plot(x, y)

    plt.title(f"Error function (learning rate = {learning_rate})")
    plt.xlabel('Iterations count', labelpad=0)
    plt.ylabel('Sum of Squares of output errors for training dataset')

    ax = plt.gca()
    ax.set_ylim([ymin, ymax])

    plt.show()


if __name__ == "__main__":
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

    main_test()
