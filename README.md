# Multilayer Perceptron Neural Network

## General
Multilayer perceptron with backpropagation algorithm, implemented from stratch. Project developed for Neural Networks in Biomedicine course at the WUT.

## Architecture
Neural network is represented by *NeuralNetwork* object and it allows to maniupalate nn, e.g. set structure, learning params, start learning/testing process based on the specified dataset. *Layer* object represents the neural network layer (hidden or output) and allows to set layer params (inputs count, neurons/outputs count, activation function, bias). Every *Layer* object has weights matrix with *m x n* with weights for each input connection in the layer (*m-th* row contains *m-th* neuron inputs and *n-th* column contains *n-th* input in the layer). 



