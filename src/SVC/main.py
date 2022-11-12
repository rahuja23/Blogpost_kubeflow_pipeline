import joblib
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import NamedTuple
import argparse
def svm_process(args) -> NamedTuple('Outputs', [('svmdir',str), ('svmscore',float)]):
    log_folder = args.log_folder
    numpy_folder = args.numpy_folder
    train_X = joblib.load(open(numpy_folder + '/train_X.pkl', 'rb'))
    test_X = joblib.load(open(numpy_folder + '/test_X.pkl', 'rb'))
    train_Y = joblib.load(open(log_folder + '/train_Y.pkl', 'rb'))
    test_Y = joblib.load(open(log_folder + '/test_Y.pkl', 'rb'))
    scaler = StandardScaler()
    train_X_s = scaler.fit(train_X).transform(train_X)

    clf = SVC(kernel='linear')
    t = clf.fit(train_X_s, np.array(train_Y).reshape(-1, 1))
    y_pred = clf.predict(test_X)
    svm_score = accuracy_score(test_Y, y_pred)

    if not os.path.isdir(numpy_folder + '/svm'):
        os.makedirs(numpy_folder + '/svm')
    svm_folder = numpy_folder + '/svm'
    joblib.dump(t, svm_folder + '/svm.pkl')
    return ([svm_folder, svm_score])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", type=str, default="")
    parser.add_argument("--numpy_folder", type=str, default="")
    args = parser.parse_args()
    return args

if __name__ =="__main__":
    args = parse_args()
    folder = svm_process(args)
