from sklearn import datasets
from data import utils as data_utils
from nn import single_layer_tf


def get_iris_data():
    iris = datasets.load_iris()
    data = iris['data']
    target = iris['target']
    return data, target


def get_processed_iris_data():
    """
    return iris data with bias column in data and columns as one hot encoded
    vectors
    :return:
    """
    data, target = get_iris_data()
    data = data_utils.append_bias(data)
    target = data_utils.to_onehot(target)
    return data, target


def classify():
    data, target = get_processed_iris_data()
    # 10 percent testing set
    train_x, train_y, test_x, test_y = data_utils.get_train_test(
        data, target, 0.1)
    single_layer_tf.run_sgd(train_x, train_y, test_x, test_y)

