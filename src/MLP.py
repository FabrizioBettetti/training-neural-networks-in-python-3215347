import numpy as np

class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By default it's 1.0."""

    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias).""" 
        
        # np.random.rand --> create an array of the given shape and
        # populate it with random samples from a uniform distribution
        # over [0, 1). In questo caso abbiamo un vettore (rango 1)
        # avente inputs+1 elementi. Moltiplicando per 2 e sottraendo
        # 1 otteniamo una distribuzione uniforme su [-1, 1)
        self.weights = (np.random.rand(inputs+1) * 2) - 1 

        self.bias = bias


    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        x_sum = np.dot(np.append(x,self.bias),self.weights)
        return self.sigmoid(x_sum)
