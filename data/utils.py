import math
import numpy as np
import tensorflow as tf


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
    length = data.shape[0]
    test_size = math.floor(length * fraction)
    train_size = length - test_size
    split = [train_size, test_size]
    train_data, test_data = tf.split(data, split, axis=0)
    train_target, test_target = tf.split(target, split, axis=1)
    return train_data, train_target, test_data, test_target



