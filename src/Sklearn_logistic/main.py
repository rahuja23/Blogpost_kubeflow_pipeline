from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
import os
import argparse
from typing import NamedTuple

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", type=str, default="")
    parser.add_argument("--numpy_folder", type=str, default="")
    args = parser.parse_args()
    return args
def sklearn_logistic(args)->NamedTuple('Outputs',[('logdir',str), ('sklearndir',str), ('sklearnscore',float)]):
    log_folder = args.log_folder
    numpy_folder = args.numpy_folder
    train_X = joblib.load(open(numpy_folder + '/train_X.pkl', 'rb'))
    test_X = joblib.load(open(numpy_folder + '/test_X.pkl', 'rb'))

    train_Y = joblib.load(open(log_folder + '/train_Y.pkl', 'rb'))
    test_Y = joblib.load(open(log_folder + '/test_Y.pkl', 'rb'))

    clf = SGDClassifier(loss='log')
    clf.fit(train_X, np.array(train_Y).reshape(-1, 1))
    y_pred = clf.predict(test_X)
    y_pred_probs = clf.predict(test_X)

    print(f"Scikit learn logistic regression accuracy is {accuracy_score(test_Y, y_pred) * 100:.2f}")

    if not os.path.isdir(numpy_folder + '/sklearn'):
        os.makedirs(numpy_folder + '/sklearn')
    sklearn_folder = numpy_folder + '/sklearn'
    joblib.dump(clf, sklearn_folder + '/sklearn.pkl')

    sklearn_score = accuracy_score(test_Y, y_pred)

    return ([log_folder, sklearn_folder, sklearn_score])

if __name__ =="__main__":
    args = parse_args()
    folder =sklearn_logistic(args)
