import numpy as np
import numpy.typing as npt
from typing import List

class Module:
    registry = []

    def forward(self, *args, **kwargs):
        """Subclasses must override this method"""
        raise NotImplementedError(
            "Subclasses must implement the forward method"
        )

    def backward(self, *args, **kwargs):
        """Subclasses must override this method"""
        raise NotImplementedError(
            "Subclasses must implement the backward method"
        )

    def __call__(self, *args, **kwargs):
        Module.registry.append(self)
        return self.forward(*args, **kwargs)

    def parameters(self):
        parameters = []
        for layer in Module.registry:
            parameters.append((layer.parameters(), layer.gradients()))
        return parameters

    def gradients(self):
        return []

    def backpropagate(self, loss_grad: npt.NDArray):
        grad = loss_grad
        for layer in reversed(Module.registry):
            grad = layer.backward(grad)


class RNN(Module):
    def __init__(self, seq_len: int, input_size: int,
                 hidden_size: int, output_size: int):
        super().__init__()
