import numpy as np
import joblib
import os
from typing import NamedTuple
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", type=str, default="")
    parser.add_argument("--numpy_folder", type=str, default="")
    args = parser.parse_args()
    return args

def sigmoid(z):
    h = 1 / (1 + np.exp(-z))
    return h

def gradientDescent(x, y, theta, alpha, num_iters, c):
    m = x.shape[0]
    for i in range(0, num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        J = (-1 / m) * ((np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h))) + (c * np.sum(theta)))
        theta = theta - (alpha / m) * np.dot((x.T), (h - y))
        J = float(J)
    return J, theta
def predict_tweet(x, theta):
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred
def logistic(args) -> NamedTuple('Outputs', [('logdir',str), ('logisticdir',str), ('logisticscore',float)]):
    log_folder = args.log_folder
    numpy_folder = args.numpy_folder
    train_X = joblib.load(open(numpy_folder + '/train_X.pkl', 'rb'))
    test_X = joblib.load(open(numpy_folder + '/test_X.pkl', 'rb'))
    train_Y = joblib.load(open(log_folder + '/train_Y.pkl', 'rb'))
    test_Y = joblib.load(open(log_folder + '/test_Y.pkl', 'rb'))
    np.random.seed(1)
    J, theta = gradientDescent(train_X, np.array(train_Y).reshape(-1, 1), np.zeros((3, 1)), 1e-7, 1000, 0.1)
    print(f"The cost after training is {J:.8f}.")
    print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")
    predicted_probs = predict_tweet(test_X, theta)
    predicted_labels = np.where(predicted_probs > 0.5, 1, 0)
    print(
        f"Own implementation of logistic regression accuracy is {len(predicted_labels[predicted_labels == np.array(test_Y).reshape(-1, 1)]) / len(test_Y) * 100:.2f}")

    if not os.path.isdir(numpy_folder + '/logistic'):
        os.makedirs(numpy_folder + '/logistic')
    logistic_folder = numpy_folder + '/logistic'
    joblib.dump(theta, logistic_folder + '/logistic.pkl')

    logistic_score = len(predicted_labels[predicted_labels == np.array(test_Y).reshape(-1, 1)]) / len(test_Y)

    return ([log_folder, logistic_folder, logistic_score])

if __name__ =="__main__":
    args = parse_args()
    folder = logistic(args)


