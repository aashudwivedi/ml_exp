# implementing simple nn with just one hidden layer, using tensorflow
import tensorflow as tf
import numpy as np


def get_weights(shape):
    return tf.random_normal(shape=shape, stddev=0.1)


def build_network(x, y, h_len=256):
    """

    :param x: data tensor
    :param y: label tensor
    :param h_len: len of the hidden layer
    :return:
    """
    x_len = x.shape[1]
    y_len = y.shape[1]

    # declare the variables in the graph
    x_var = tf.placeholder("float", shape=x_len)
    y_var = tf.placeholder("float", shape=y_len)

    w1 = tf.Variable(get_weights((x_len, h_len)))
    w2 = tf.Variable(get_weights((h_len, x_len)))

    # build the computation graph for forward pass
    h = tf.nn.sigmoid(tf.matmul(x_var, w1))
    probs = tf.matmul(h, w2)
    predict = tf.argmax(probs, axis=1)

    # build computation graph for backward pass
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=predict,
        logits=probs))
    updates = tf.train.GradientDescentOptimizer(00.1).minimize(cost)
    return x_var, y_var, predict, updates


def get_accuracies(session, predict, x, y, x_val, y_val):
    predictions = session.run(predict, feed_dict={
        x: x_val,
        y: y_val,
    })
    error = predictions == y_val
    return np.mean(error)


def run_sgd(train_x, train_y, test_x, test_y):
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)
    session.close()

    x, y, predict, update = build_network(train_x, train_y)

    for epoch in range(100):
        for i in range(train_x.shape[0]):
            session.run(update, feed_dict={
                x: train_x[i: i + 1],
                y: train_y[i: i + 1]
            })
        train_acc = get_accuracies(session, predict, x, y, train_x, train_y)
        test_acc = get_accuracies(session, predict, x, y, test_x, test_y)
        print(train_acc, test_acc)
    session.close()
