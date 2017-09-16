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
    x_cols = x.shape[1]
    y_cols = y.shape[1]

    # declare the variables in the graph
    x_var = tf.placeholder("float", shape=(None, x_cols))
    y_var = tf.placeholder("float", shape=(None, y_cols))

    w1 = tf.Variable(get_weights((x_cols, h_len)))
    w2 = tf.Variable(get_weights((h_len, y_cols)))

    # build the computation graph for forward pass
    print('shape of x_var = {}'.format(tf.shape(x_var)))
    print('shape of w1 = {}'.format(tf.shape(w1)))

    h = tf.nn.sigmoid(tf.matmul(x_var, w1))
    probs = tf.matmul(h, w2)
    predict = tf.argmax(probs, axis=1)

    print('shape of y_var = {}'.format(tf.shape(y_var)))
    print('shape of probs = {}'.format(tf.shape(probs)))

    # build computation graph for backward pass
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y_var,
        logits=probs))
    updates = tf.train.GradientDescentOptimizer(00.1).minimize(cost)
    return x_var, y_var, predict, updates


def get_accuracies(session, predict, x, y, x_val, y_val):
    predictions = session.run(predict, feed_dict={
        x: x_val,
        y: y_val,
    })

    error = predictions == np.argmax(y_val, axis=1)
    return np.mean(error)


def run_sgd(train_x, train_y, test_x, test_y):
    x, y, predict, update = build_network(train_x, train_y)

    print('shape of train x {}'.format(train_x.shape))
    print('shape of test x {}'.format(test_x.shape))
    print('shape of train y {}'.format(train_y.shape))
    print('shape of test y {}'.format(test_y.shape))

    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)

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
