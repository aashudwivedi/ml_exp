import numpy as np
from sklearn.model_selection import train_test_split


def append_bias(data):
    """ adds bias column of 1s to the data"""
    row, col = data.shape
    ones = np.ones((row, col + 1))
    ones[:, :-1] = data
    return ones


def to_onehot(targets):
    unique_labels = len(np.unique(targets))
    return np.eye(unique_labels)[targets]


def get_train_test(data, target, fraction):
    """ data and target length should be same"""
    # TODO: perform the test train split inside the tensorflow graph
    return train_test_split(data, target, test_size=fraction,)



