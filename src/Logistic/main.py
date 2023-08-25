<<<<<<< HEAD
import mlflow
=======
>>>>>>> 9643c8dbf8c807e406368a719a7a61d4b9b6399e
import numpy as np
import joblib
import os
from typing import NamedTuple
import argparse
<<<<<<< HEAD
import shutil
from dotenv import load_dotenv
load_dotenv()
os.getenv("AWS_ACCESS_KEY_ID")
os.getenv("AWS_SECRET_ACCESS_KEY")
os.getenv("MLFLOW_S3_ENDPOINT_URL")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--tracking_uri", type=str, default="http://127.0.0.1:5001")
    parser.add_argument("--experiment_id", type=str, default=1)
    parser.add_argument("--run_id", type= str, default="")
=======

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", type=str, default="")
    parser.add_argument("--numpy_folder", type=str, default="")
>>>>>>> 9643c8dbf8c807e406368a719a7a61d4b9b6399e
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
<<<<<<< HEAD
def logistic(args):
    load_dotenv()
    os.getenv("AWS_ACCESS_KEY_ID")
    os.getenv("AWS_SECRET_ACCESS_KEY")
    os.getenv("MLFLOW_S3_ENDPOINT_URL")
    log_folder = args["log_folder"]
    output_dir = args["output_dir"]
    experiment_id = args["experiment_id"]
    run_id = args["run_id"]
    numpy_folder = log_folder + '/numpy'
    numpy_folder = log_folder + '/numpy'
    print("EXPERIMENT_ID:", args["experiment_id"])
    print("TRACKING_URI:", args["tracking_uri"])
    tracking_uri = args["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    print("******************CREATING CLIENT******************************")
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    print("************************Existing RUN**************************")
    print("RUN ID:", run_id)
    with mlflow.start_run(run_id=run_id) as run:
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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        files = os.listdir(log_folder)
        metric_dict = {}
        metric_dict["logistic_score"] = logistic_score
        mlflow.log_metrics(metric_dict)
        shutil.copytree(logistic_folder, output_dir, dirs_exist_ok=True)
        mlflow.log_artifact(output_dir, "logistic_output")
        file1 = open(os.path.join(output_dir, "logistic_score.txt"), "w")
        file1.write(str(logistic_score))
        file1.close()

if __name__ =="__main__":
    args = parse_args()
    args = vars(args)
    os.getenv("AWS_ACCESS_KEY_ID")
    os.getenv("AWS_SECRET_ACCESS_KEY")
    os.getenv("MLFLOW_S3_ENDPOINT_URL")
    logistic(args)
=======
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
>>>>>>> 9643c8dbf8c807e406368a719a7a61d4b9b6399e


