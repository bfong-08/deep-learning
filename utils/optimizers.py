import numpy as np
import numpy.typing as npt
from typing import List
from utils.components import Module

class SGD(Module):
    """
    The simplest form of gradient descent

    Args:
        lr: The learning rate that the gradients
        of the parameters is multiplied by
    """
    def __init__(self, parameters: List[npt.NDArray], lr: float=0.01):
        super().__init__()
        self.learning_rate = lr
        self.parameters = parameters

    def step(self):
        for param, grad in self.parameters:
            param -= self.learning_rate * grad

