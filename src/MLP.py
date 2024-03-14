import numpy as np

class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By default it's 1.0."""

    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias).""" 
        self.weights = (np.random.rand(inputs+1) * 2) - 1 
        self.bias = bias

    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        x_sum = np.dot(np.append(x,self.bias),self.weights)
        return self.sigmoid(x_sum)

    def set_weights(self, w_init):
        """Set the weights. w_init is a python list with the weights."""
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        """Evaluate the sigmoid function for the floating point input x."""
        return 1/(1+np.exp(-x))



class MultiLayerPerceptron:     
    """A multilayer perceptron class that uses the Perceptron class above.
       Attributes:
          layers:  A python list with the number of neurons per layer.
          bias:    The bias term. The same bias is used for all neurons.
          eta:     The learning rate."""

    def __init__(self, layers, bias = 1.0):
        """Return a new MLP object with the specified parameters.""" 
        self.layers = np.array(layers,dtype=object)
        self.bias = bias
        self.network = [] # The list of lists of neurons
        self.values = []  # The list of lists of output values.

        for i in range(len(self.layers)):   # Per ogni layer
            self.values.append([])
            self.network.append([])

            # self.values[i] identifica la lista di valori in uscita dal layer i-esimo.
            # Tale lista presenta un numero di elementi pari al numero di neuroni del
            # layer i-esimo. Inizialmente i valori delle uscite di ciascun layer sono a 0.0
            self.values[i] = [0.0 for j in range(self.layers[i])]

            if i > 0:      #network[0] is the input layer, so it has no neurons
                for j in range(self.layers[i]):  # Per ogni neurone del layer i-esimo
                    # self.network[i] individua la lista di neuroni contenuti nel layer i-esimo.
                    # A questa lista appendo un percettrone avente un numero di ingressi pari al
                    # numero di neuroni presenti nel layer precedente
                    self.network[i].append(Perceptron(inputs = self.layers[i-1], bias = self.bias))
        
        # La lista di liste viene convertita in numpy array di numpy array 
        self.network = np.array([np.array(x) for x in self.network],dtype=object)
        self.values = np.array([np.array(x) for x in self.values],dtype=object)
