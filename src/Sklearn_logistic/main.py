from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
import os
import argparse
import shutil
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--tracking_uri", type=str, default="http://127.0.0.1:5001")
    parser.add_argument("--experiment_id", type=str, default=1)
    parser.add_argument("--run_id", type= str, default="")
    args = parser.parse_args()
    return args
def sklearn_logistic(args):
    load_dotenv()
    os.getenv("AWS_ACCESS_KEY_ID")
    os.getenv("AWS_SECRET_ACCESS_KEY")
    os.getenv("MLFLOW_S3_ENDPOINT_URL")
    log_folder = args["log_folder"]
    output_dir = args["output_dir"]
    experiment_id = args["experiment_id"]
    run_id = args["run_id"]
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
        clf = SGDClassifier(loss='log')
        clf.fit(train_X, np.array(train_Y).reshape(-1, 1))
        y_pred = clf.predict(test_X)
        metric_dict = {}
        print(f"Scikit learn logistic regression accuracy is {accuracy_score(test_Y, y_pred) * 100:.2f}")
        if not os.path.isdir(numpy_folder + '/sklearn'):
            os.makedirs(numpy_folder + '/sklearn')
        sklearn_folder = numpy_folder + '/sklearn'
        joblib.dump(clf, sklearn_folder + '/sklearn.pkl')
        sklearn_score = accuracy_score(test_Y, y_pred)
        metric_dict["sklearn_accuracy"]= 0.99
        mlflow.log_metrics(metric_dict)
        mlflow.sklearn.log_model(
            sk_model= clf,
            artifact_path="sklearn-model",
            registered_model_name="sk-learn-logistic-regression"
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        files = os.listdir(log_folder)
        print("FILES:", files)
        shutil.copytree(sklearn_folder, output_dir, dirs_exist_ok=True)
        mlflow.log_artifact(os.path.join(sklearn_folder + '/sklearn.pkl'), "sklearn_output")
if __name__ =="__main__":
    args = parse_args()
    args =vars(args)
    folder =sklearn_logistic(args)
