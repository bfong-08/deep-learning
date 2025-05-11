import numpy as np
import numpy.typing as npt
from typing import Callable

import utils.init_weights
from utils.components import Module


class FCLayer(Module):
    """
    Represents a single layer of a neural network fully
    connected by weights in the form f(Wx+b)

    Args:
        input_size: the size of each input sample
        output_size: the size of each output sample
        activation: the function to be applied
        to the resulting tensor after adding bias
        bias: determines whether additive bias will
        be added to the product of the inputs and
        weights
            Default: ``True``
        init_weights: the function used to determine
        the initialization of the weight matrix
            Default: ``utils.init_weights.uniform``

    Attributes:
        weights: the learnable weights of the layer
    """

    weights: npt.NDArray

    def __init__(self, input_size: int, output_size: int,
                 activation: Callable[[npt.NDArray],
                 npt.NDArray]=None, bias:bool=True,
                 init_weights: Callable[[int, int],
                 npt.NDArray]=utils.init_weights.uniform) -> None:
        super().__init__()

        self.activation = activation
        self.weights = init_weights(input_size, output_size)

        self.input_cache = None
        self.weight_grad = None
        self.bias_grad = None

        if bias: self.bias = np.zeros(output_size)
        else: self.bias = None

    def forward(self, x: npt.NDArray):
        """
        Args:
            x: the input sample

        Returns: the output vector after
        applying the activation function
        """
        self.input_cache = x

        out = np.dot(x, self.weights)

        if self.bias is not None: out += self.bias
        if self.activation is not None: out = self.activation(out)
        return out

    def backward(self, grad_output: npt.NDArray,
                 optimizer: Callable[[npt.NDArray], npt.NDArray]):
        self.weight_grad = np.matmul(np.expand_dims(self.input_cache, -1),
                                np.expand_dims(grad_output, -2))
        self.bias_grad = grad_output
        next_grad = np.dot(grad_output, self.weights.T)
        return next_grad

    def parameters(self):
        return [self.weights, self.bias]

    def gradients(self):
        return [self.weight_grad, self.bias_grad]