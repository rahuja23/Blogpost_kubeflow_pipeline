import numpy as np
import joblib
import os
from typing import NamedTuple
import argparse
def extract_features(processed_tweet, bow_word_frequency):
    features = np.zeros((1, 3))
    features[0, 0] = 1

    for word in processed_tweet:
        features[0, 1] = bow_word_frequency.get((word, 1), 0) + features[0, 1]
        features[0, 2] = bow_word_frequency.get((word, 0), 0) + features[0, 2]
    return features

def numpy_process(args) -> NamedTuple('Outputs', [('logdir',str), ('numpydir',str)]):
    log_folder = args.log_folder
    bow_word_frequency = joblib.load(open(log_folder + '/bow_word_frequency.pkl', 'rb'))
    train_X_tweet = joblib.load(open(log_folder + '/train_X_tweet.pkl', 'rb'))
    test_X_tweet = joblib.load(open(log_folder + '/test_X_tweet.pkl', 'rb'))
    train_Y = joblib.load(open(log_folder + '/train_Y.pkl', 'rb'))
    test_Y = joblib.load(open(log_folder + '/test_Y.pkl', 'rb'))

    train_X = np.zeros((len(train_X_tweet), 3))
    for index, tweet in enumerate(train_X_tweet):
        train_X[index, :] = extract_features(tweet, bow_word_frequency)

    test_X = np.zeros((len(test_X_tweet), 3))
    for index, tweet in enumerate(test_X_tweet):
        test_X[index, :] = extract_features(tweet, bow_word_frequency)

    print(f"train_X {train_X.shape}, test_X {test_X.shape}")

    if not os.path.isdir(log_folder + '/numpy'):
        os.makedirs(log_folder + '/numpy')

    numpy_folder = log_folder + '/numpy'

    joblib.dump(train_X, numpy_folder + '/train_X.pkl')
    joblib.dump(test_X, numpy_folder + '/test_X.pkl')
    joblib.dump(train_Y, numpy_folder + "/train_Y.pkl")
    joblib.dump(test_Y, log_folder + "/test_Y.pkl")
    return ([log_folder, numpy_folder])
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", type=str, default="")
    args = parser.parse_args()
    return args
if __name__ =="__main__":
    args = parse_args()
    folder = numpy_process(args)
