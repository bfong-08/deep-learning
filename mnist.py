from utils.layers import FCLayer
from utils.components import Module
from utils.activations import Sigmoid
from utils.datasets import batch_dataset
from utils.loss import CrossEntropyLoss
from utils.optimizers import SGD
from utils.init_weights import xavier_normal

import pandas as pd
import numpy as np

np.set_printoptions(precision=4, suppress=True)

dataframe = pd.read_csv('./datasets/mnist/mnist_train.csv')
labels = dataframe['label'].to_numpy()
labels = np.eye(max(labels + 1))[labels]
inputs = dataframe.drop(columns=['label']).to_numpy()
inputs = inputs / np.max(inputs)

batch_size = 128
inputs, labels = batch_dataset(inputs, labels, batch_size)

class Model(Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = FCLayer(input_size, hidden_size, Sigmoid(), init_weights=xavier_normal)
        self.fc2 = FCLayer(hidden_size, output_size)

    def __call__(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


input_size = inputs.shape[-1]
hidden_size = 64
output_size = labels.shape[-1]

model = Model(input_size, hidden_size, output_size)
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters())

for epoch in range(10):
    for j, (x, y) in enumerate(zip(inputs, labels)):
        predicted = model(x)
        loss = criterion(predicted, y)
        if j % 100 == 0: print(f"loss: {loss:.4f}")
        criterion.backward()
        optimizer.step()
    print("-" * 30)
