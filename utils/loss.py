import numpy as np
import numpy.typing as npt
from utils.components import Module

class CrossEntropyLoss(Module):
    """
    Also known as log loss, used to measure
    the performance of classification models.
    Normalizes the inputted tensor using
    softmax before calculating the loss.
    """

    def __init__(self):
        super().__init__()
        self.label_cache: npt.NDArray | None = None
        self.softmax_cache: npt.NDArray | None = None

    def forward(self, pred: npt.NDArray,
                actual: npt.NDArray):
        """
        Args:
            pred: The predicted output tensor from the model
            actual: The true one-hot encoded output tensor
        Returns: Loss
        """
        self.label_cache = actual

        exp = np.exp(pred - np.max(pred, axis=-1, keepdims=True))
        softmaxed = exp / np.sum(exp, axis=-1, keepdims=True)

        self.softmax_cache = softmaxed

        loss = -np.sum(np.multiply(actual, np.log(softmaxed + 1e-15)), axis=-1)
        loss = np.sum(loss) / loss.shape[0]
        return loss

    def backward(self, *args):
        next_grad = self.softmax_cache - self.label_cache
        return next_grad
