import numpy as np

def zeros(input_size: int, output_size: int):
    """
    Args:
        input_size: the size of each input sample
        output_size: the size of each output sample

    Returns: the initial weight matrix where each weight
    is zero
    """
    return np.zeros((input_size, output_size))

def uniform(input_size: int, output_size: int):
    """
    Args:
        input_size: the size of each input sample
        output_size: the size of each output sample

    Returns: the initial weight matrix where each weight
    is a random number between -1 and 1
    """
    return np.random.uniform(-1, 1, (input_size, output_size))

def normal(input_size: int, output_size: int):
    """
        Args:
            input_size: the size of each input sample
            output_size: the size of each output sample

        Returns: the initial weight matrix where each weight
        is chosen from a normal distribution
        """
    return np.random.normal(0, 1, (input_size, output_size))


def he_normal(input_size: int, output_size: int):
    """
    Suited for layers with non-Saturating
    activation functions like ReLU. Also
    called kaiming activation
    Args:
        input_size: the size of each input sample
        output_size: the size of each output sample

    Returns: the initial weight matrix where each weight
    is taken from a normal distribution with a variance
    of :math:`\sqrt{2/(n_in)}`
    """
    std = np.sqrt(2 / (input_size + output_size))
    return np.random.normal(0, std, size=(input_size, output_size))

def he_uniform(input_size: int, output_size: int):
    """
    Suited for layers with non-Saturating
    activation functions like ReLU. Also
    called kaiming activation
    Args:
        input_size: the size of each input sample
        output_size: the size of each output sample

    Returns: the initial weight matrix where each weight
    is taken from a uniform distribution with a
    scaling factor of :math:`\sqrt{6/(n_in)}`
    """
    limit = np.sqrt(6 / input_size)
    return np.random.uniform(-limit, limit, size=(input_size, output_size))

def xavier_normal(input_size: int, output_size: int):
    """
    Suited for saturating functions like sigmoid
    or tanh. Also called glorot initialization
    Args:
        input_size: the size of each input sample
        output_size: the size of each output sample

    Returns: the initial weight matrix where each weight
    is taken from a normal distribution with a variance
    of :math:`\sqrt{2/(n_in + n_out)}`
    """
    std = np.sqrt(2 / (input_size + output_size))
    return np.random.normal(0, std, (input_size, output_size))

def xavier_uniform(input_size: int, output_size: int):
    """
    Suited for saturating functions like sigmoid
    or tanh. Also called glorot initialization
    Args:
        input_size: the size of each input sample
        output_size: the size of each output sample

    Returns: the initial weight matrix where each weight
    is taken from a uniform distribution with a
    scaling factor of :math:`\sqrt{6/(n_in + n_out)}`
    """
    limit = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-limit, limit, (input_size, output_size))
