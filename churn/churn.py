import os
import numpy as np
import pandas as pd

from keras import models, layers, utils
from sklearn.model_selection import train_test_split
from sklearn import metrics

def get_data():
    cd = os.getcwd()
    path = os.path.join(os.getcwd(), 'data/train.csv')
    return pd.read_csv(path)


def prepare_data(df):
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling']
    cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

    data = df[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']]

    data.loc[:, 'TotalCharges'] = pd.to_numeric(
        data['TotalCharges'], errors='coerce')
    data.loc[data['TotalCharges'].isna(), 'TotalCharges'] = 0

    frames = [data]

    for col in binary_cols:
        frames.append(pd.get_dummies(df[col], prefix=col, drop_first=True))

    for col in cat_cols:
        frames.append(pd.get_dummies(df[col], prefix=col))

    data = pd.concat(frames, axis=1)
    y = pd.get_dummies(df['Churn'], drop_first=True)

    return data, y


def get_test_train(test_size=0.3):
    df = get_data()
    X, Y = prepare_data(df)
    return train_test_split(X, Y, test_size=test_size)


def print_metrics(y_true, y_pred):
    print('-' * 10)
    print('y_true classes:')
    total = y_true.shape[0]
    true_count = np.sum(y_true)
    false_count = total - true_count
    print('total={}, true={}, false={}'.format(total, true_count, false_count))
    print('accuracy={}'.format(metrics.accuracy_score(y_true, y_pred)))
    print('f1 score ={}'.format(metrics.f1_score(y_true, y_pred)))
    print(metrics.classification_report(y_true, y_pred))


def main(train_x, train_y):
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_dim=100))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=10, batch_size=32)








