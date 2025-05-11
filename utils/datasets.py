import numpy as np
import numpy.typing as npt

def batch_dataset(inputs:npt.NDArray, labels:npt.NDArray,
                  batch_size: int, shuffle:bool=True) \
        -> (npt.NDArray, npt.NDArray):
    """
    Splits a dataset of inputs and labels
    into batches

    Args:
        inputs: All input samples
        labels: All labels
        batch_size: The size of each batch that
        the data will be split into
        shuffle: Determines if the data will
        be shuffled prior to batching
            Default: ``True``

    Returns: The batched inputs and labels
    """

    len_dataset = inputs.shape[0]
    remainder = len_dataset % batch_size
    inputs, labels = inputs[:len_dataset-remainder], labels[:len_dataset-remainder]

    if shuffle:
        indices = np.arange(len_dataset - remainder)
        np.random.shuffle(indices)
        inputs = inputs[indices]
        labels = labels[indices]

    inputs = np.array_split(inputs, (len_dataset - remainder) // batch_size)
    labels = np.array_split(labels, (len_dataset - remainder) // batch_size)

    return np.array(inputs), np.array(labels)
