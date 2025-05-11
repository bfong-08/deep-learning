import numpy as np
import numpy.typing as npt
from utils.components import Module

class ReLU(Module):
    """
    Applies the Rectified Linear Unit function
    to a tensor
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: npt.NDArray):
        return np.where(x > 0, x, 0)

class Sigmoid(Module):
    """
    Applies the Sigmoid function to a tensor
    """

    def __init__(self):
        self.output_cache = None

    def forward(self, z: npt.NDArray):
        out = 1 / (1 + np.exp(-z))
        self.output_cache = out
        return out

    def backward(self, prev_grad, *args):
        grad = np.multiply(self.output_cache, (1 - self.output_cache))
        return np.multiply(grad, prev_grad)



class Tanh(Module):
    """
    Applies the Tanh (Hyperbolic Tangent) function
    to a tensor
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: npt.NDArray):
        exp = np.exp(x)
        n_exp = np.exp(-x)
        return (exp - n_exp) / (exp + n_exp)

class Softmax(Module):
    """
    Converts a tensor of outputs to a probability
    distribution of those outputs. Usually used as
    the last activation function alongside Cross
    Entropy Loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: npt.NDArray):
        self.input_cache = z

        exp = np.exp(z)
        out = exp / np.sum(exp)

        return out

