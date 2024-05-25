import numpy as np

class Layer:
    """
    Class that represents single hidden/output layer.

    In further descriptions 'vector' means 'row vector' (nx1 size matrix)

    Attributes
    -----------
    weights_matrix
        Weights matrix for the layer, where each row represents weights of single neuron in the layer,
        while each column represents weights of specified layer input.


                               BIAS         IN_1       IN_2     ...     IN_n
                         ---                                                     ---
        NEURON_1        |      w_1_0        w_1_1      w_1_2    ...     w_1_n       |
                        |                                                           |
        NEURON_2        |      w_2_0        w_2_1      w_2_2    ...     w_2_n       |
                        |                                                           |
          ...           |       ...         ...        ...      ...      ...        |
                        |                                                           |
        NEURON_m        |      w_3_0        w_m_1      w_m_2    ...     w_m_n       |
                         ---                                                     ---

    input_data
        Network input vector (for single record) or matrix (every row represents single record).

    preoutput_data
        Vector of the computed inputs sums for each neuron, which will be passed to the activation function
        (matrix of the computed inputs for multiple records input data; similarly to input_data property)

    activation_function
        Activation function for each neuron in the layer.

    activation_derivative
        Activation function derivative for each neuron in the layer.
    """

    weights_matrix = None
    input_data = None
    preoutput_data = None

    activation_function = None
    activation_derivative = None

    SIGMOID_FUNCTION = lambda input_vector: 1 / (1 + np.exp(-input_vector))
    SIGMOID_DERIVATIVE = lambda input_vector: (1 / (1 + np.exp(-input_vector))) * (1 - 1 / (1 + np.exp(-input_vector)))

    RELU_FUNCTION = lambda input_vector: np.maximum(np.zeros(input_vector.shape), input_vector)
    RELU_DERIVATIVE = lambda input_vector: (input_vector > 0).astype(int)

    BIAS = 1



    def __init__(self,
                 input_size,
                 output_size,
                 activation_foo=SIGMOID_FUNCTION,
                 derivative_foo=SIGMOID_DERIVATIVE,
                 weights_random_range=(-0.2, 0.2)
                 ):

        # weights random initialization
        x0, x1 = weights_random_range
        self.weights_matrix = np.random.rand(output_size, input_size + 1) * (x1-x0) - abs(x0)
        self.activation_function = activation_foo
        self.activation_derivative = derivative_foo


    def compute_output(self, input_data):
        """Computing neurons outputs vector (matrix) for specified input_data vector (matrix)."""

        input_data = np.atleast_2d(input_data)
        self.input_data = np.column_stack((np.ones(input_data.shape[0]) * self.BIAS, input_data))
        self.preoutput_data = ( self.weights_matrix @ self.input_data.T ).T
        return self.activation_function(self.preoutput_data)


    def get_no_bias_weights_matrix(self):
        """Returns weights matrix without bias weights (i.e. weights matrix without first column)."""
        return self.weights_matrix[:, 1:]

    def compute_preoutput_derivatives(self):
        """Computing activation function derivative values vector for previously computed preoutput data.
        It will be useful in backpropagation algorithm."""
        return self.activation_derivative(self.preoutput_data)


    def update_weights(self, d_weights):
        """Add specified d_weights matrix to current layer weights_matrix; d_weights size must be the same as weights_matrix size"""
        self.weights_matrix += d_weights








class NeuralNetwork:
    """
    Class that represents multilayer perceptron nn (MLP) based on backpropagation algorithm.


    Attributes
    -----------
    network_input
        Current network input (input vector selected from training dataset (learning) or input matrix (predicting))

    hidden_layers
        List of hidden layers in the neural network.

    output_layer
        Output layer of the neural network.

    learning_rate
        Learning rate used in the backpropagation algorithm.
    """

    network_input = None
    hidden_layers = None
    output_layer: Layer

    learning_rate = None


    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.2):
        """
        :param input_size:
            Size of the nn input layer.

        :param hidden_sizes:
            Array of the neurons counts in the following hidden layers
            (e.g. [2, 4] means 2 hidden layers - first with 2 neurons and second with 4 neurons)
        """

        self.set_structure(input_size, hidden_sizes, output_size)
        self.set_learning_rate(learning_rate)


    def set_structure(self, input_size, hidden_sizes, output_size):
        """Set structure of the neural network."""

        self.hidden_layers = list()

        cols_cnt = input_size
        for rows_cnt in hidden_sizes:
            self.hidden_layers.append(
                Layer(cols_cnt, rows_cnt, activation_foo=Layer.RELU_FUNCTION, derivative_foo=Layer.RELU_DERIVATIVE)
            )
            cols_cnt = rows_cnt

        self.output_layer = Layer(cols_cnt, output_size, activation_foo=Layer.SIGMOID_FUNCTION,
                                  derivative_foo=Layer.SIGMOID_DERIVATIVE)


    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate



    def learn_network(self, learning_data: np.ndarray, iterations_cnt=3000, sequentially=False):
        """Run learning process.

        :param learning_data:
            learning dataset

        :param iterations_cnt:
            iterations counter for learning process

        :param sequentially:
            Specify policy for selecting learning vectors in the following iterations. If sequentially True, then
            learning vectors will be selected sequentially row by row from the learning dataset; otherwise every learning
            vector will be randomly chosen from the learning dataset.

        :returns error_function:
            Vector of the error function values (contains values for each iteration of the learning process).
            """

        error_function = np.zeros(iterations_cnt)

        for i in range(iterations_cnt):
            row_idx = i % learning_data.shape[0] if sequentially else np.random.randint(0, learning_data.shape[0]-1)
            input_vector = learning_data[row_idx, :]
            reference_output = input_vector[-1]

            network_output_vector = self.__pass_forward(input_vector[0:-1])
            network_err_vector = reference_output - network_output_vector
            self.__pass_backward(network_err_vector)

            predicted_data = self.predict(learning_data[:, 0:-1])
            threshold = np.median(predicted_data)
            formatted_predicted_data = (predicted_data >= threshold).astype(int).squeeze()
            error_function[i] = np.sum(np.square(formatted_predicted_data - learning_data[:, -1]))

        return error_function



    def __pass_forward(self, network_input):
        """Compute nn output for specified nn input (network_input can be vector for the learning process
        or matrix for the predicting process """
        layer_input = network_input
        for layer in self.hidden_layers:
            layer_input = layer.compute_output(layer_input)

        return self.output_layer.compute_output(layer_input)


    def __pass_backward(self, network_err_vector):
        """Backpropagation algorithm implementation"""

        last_layer = self.output_layer

        last_err_vector = network_err_vector
        last_err_signals_vector = last_err_vector * last_layer.compute_preoutput_derivatives()
        last_d_weights = self.__compute_d_weights_matrix(last_err_signals_vector, last_layer.input_data)

        for hidden_layer in reversed(self.hidden_layers):
            hidden_err_vector = np.atleast_2d(last_err_signals_vector) @ np.atleast_2d(last_layer.get_no_bias_weights_matrix())
            hidden_err_signals_vector = hidden_err_vector * hidden_layer.compute_preoutput_derivatives()
            hidden_d_weights = self.__compute_d_weights_matrix(hidden_err_signals_vector, hidden_layer.input_data)

            last_layer.update_weights(last_d_weights)

            last_layer = hidden_layer
            last_err_signals_vector = hidden_err_signals_vector
            last_d_weights = hidden_d_weights

        last_layer.update_weights(last_d_weights)


    def __compute_d_weights_matrix(self, err_signals_vector, input_vector):
        """Computing weights matrix delta for the layer by specifying layer signals vector and input vector. """
        return self.learning_rate * ( np.atleast_2d(err_signals_vector).T @ np.atleast_2d(input_vector) )


    def predict(self, network_input):
        """Predicting outputs for specified network_input."""
        return self.__pass_forward(network_input)
