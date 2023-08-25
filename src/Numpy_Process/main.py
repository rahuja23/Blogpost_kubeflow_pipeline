import numpy as np
import joblib
import os
from typing import NamedTuple
import argparse
<<<<<<< HEAD
import shutil
import mlflow
from dotenv import load_dotenv
load_dotenv()
os.getenv("AWS_ACCESS_KEY_ID")
os.getenv("AWS_SECRET_ACCESS_KEY")
os.getenv("MLFLOW_S3_ENDPOINT_URL")
def extract_features(processed_tweet, bow_word_frequency):
    features = np.zeros((1, 3))
    features[0, 0] = 1
=======
def extract_features(processed_tweet, bow_word_frequency):
    features = np.zeros((1, 3))
    features[0, 0] = 1

>>>>>>> 9643c8dbf8c807e406368a719a7a61d4b9b6399e
    for word in processed_tweet:
        features[0, 1] = bow_word_frequency.get((word, 1), 0) + features[0, 1]
        features[0, 2] = bow_word_frequency.get((word, 0), 0) + features[0, 2]
    return features

<<<<<<< HEAD
def numpy_process(args):
    log_folder = args["log_folder"]
    output_dir = args["output_dir"]
    print("LOG_FOLDER:", log_folder)
    print("EXPERIMENT_ID:", args["experiment_id"])
    print("RUN_ID:", args["run_id"])
    print("TRACKING_URI:", args["tracking_uri"])
    tracking_uri = args["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = args["experiment_id"]
    run_id = args["run_id"]
    with mlflow.start_run(experiment_id=experiment_id, run_id=run_id) as run:
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
        print("NUMPY_FOLDER:", numpy_folder)
        joblib.dump(train_X, numpy_folder + '/train_X.pkl')
        joblib.dump(test_X, numpy_folder + '/test_X.pkl')
        joblib.dump(train_Y, numpy_folder + "/train_Y.pkl")
        joblib.dump(test_Y, numpy_folder + "/test_Y.pkl")
        mlflow.log_artifact(os.path.join(numpy_folder, 'train_X.pkl'), "Training_data")
        mlflow.log_artifact(os.path.join(numpy_folder, 'test_X.pkl'), "Testing_data")
        mlflow.log_artifact(os.path.join(numpy_folder, 'train_Y.pkl'), "Training_labels")
        mlflow.log_artifact(os.path.join(numpy_folder, 'test_Y.pkl'), "Testing_labels")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        files  = os.listdir(log_folder)
        print("FILES:", files)
        shutil.copytree(log_folder, output_dir, dirs_exist_ok=True)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--tracking_uri", type=str, default="http://127.0.0.1:5001")
    parser.add_argument("--experiment_id", type=str, default=1)
=======
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
>>>>>>> 9643c8dbf8c807e406368a719a7a61d4b9b6399e
    args = parser.parse_args()
    return args
if __name__ =="__main__":
    args = parse_args()
<<<<<<< HEAD
    args= vars(args)
=======
>>>>>>> 9643c8dbf8c807e406368a719a7a61d4b9b6399e
    folder = numpy_process(args)
