import joblib
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
<<<<<<< HEAD
import shutil
import argparse
from dotenv import load_dotenv
import mlflow
load_dotenv()
os.getenv("AWS_ACCESS_KEY_ID")
os.getenv("AWS_SECRET_ACCESS_KEY")
os.getenv("MLFLOW_S3_ENDPOINT_URL")
def svm_process(args):
    load_dotenv()
    os.getenv("AWS_ACCESS_KEY_ID")
    os.getenv("AWS_SECRET_ACCESS_KEY")
    os.getenv("MLFLOW_S3_ENDPOINT_URL")
    log_folder = args["log_folder"]
    output_dir = args["output_dir"]
    numpy_folder = log_folder + '/numpy'
    experiment_id = args["experiment_id"]
    run_id = args["run_id"]
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
        scaler = StandardScaler()
        train_X_s = scaler.fit(train_X).transform(train_X)
        clf = SVC(kernel='linear')
        t = clf.fit(train_X_s, np.array(train_Y).reshape(-1, 1))
        y_pred = clf.predict(test_X)
        svm_score = accuracy_score(test_Y, y_pred)
        metric_dict = {}
        metric_dict["svm_score"] = svm_score
        mlflow.log_metrics(metric_dict)
        if not os.path.isdir(numpy_folder + '/svm'):
            os.makedirs(numpy_folder + '/svm')
        svm_folder = numpy_folder + '/svm'
        joblib.dump(t, svm_folder + '/svm.pkl')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        files = os.listdir(log_folder)
        print("FILES:", files)
        shutil.copytree(svm_folder, output_dir, dirs_exist_ok=True)
        file1 = open(os.path.join(output_dir, "svm_score.txt"), "w")
        file1.write(str(svm_score))
        file1.close()
        mlflow.log_artifacts(output_dir, "svm_output")
=======
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
>>>>>>> 9643c8dbf8c807e406368a719a7a61d4b9b6399e

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", type=str, default="")
<<<<<<< HEAD
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--tracking_uri", type=str, default="http://127.0.0.1:5001")
    parser.add_argument("--experiment_id", type=str, default=1)
    parser.add_argument("--run_id", type=str, default="")
=======
    parser.add_argument("--numpy_folder", type=str, default="")
>>>>>>> 9643c8dbf8c807e406368a719a7a61d4b9b6399e
    args = parser.parse_args()
    return args

if __name__ =="__main__":
    args = parse_args()
<<<<<<< HEAD
    args = vars(args)
    svm_process(args)
=======
    folder = svm_process(args)
>>>>>>> 9643c8dbf8c807e406368a719a7a61d4b9b6399e
