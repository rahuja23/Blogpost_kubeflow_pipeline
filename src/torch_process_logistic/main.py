import mlflow
import torch
import joblib
import os
import argparse
import shutil
from dotenv import load_dotenv
load_dotenv()
os.getenv("AWS_ACCESS_KEY_ID")
os.getenv("AWS_SECRET_ACCESS_KEY")
os.getenv("MLFLOW_S3_ENDPOINT_URL")
def extract_features(processed_tweet, bow_word_frequency):
    features = torch.zeros((1, 3))
    features[0, 0] = 1

    for word in processed_tweet:
        features[0, 1] = bow_word_frequency.get((word, 1), 0) + features[0, 1]
        features[0, 2] = bow_word_frequency.get((word, 0), 0) + features[0, 2]
    return features

def sigmoid(z):
    h = 1 / (1 + torch.exp(-z))
    return h

def gradientDescent(x, y, theta, alpha, num_iters, c):

    m = x.shape[0]

    for i in range(0, num_iters):
        z = torch.mm(x, theta)
        h = sigmoid(z)
        J = (-1 / m) * ((torch.mm(y.T, torch.log(h)) + torch.mm((1 - y).T, torch.log(1 - h)))
                        + (c * torch.sum(theta)))
        theta = theta - (alpha / m) * torch.mm((x.T), (h - y))
        J = float(J)
    return J, theta


def predict_tweet(x, theta):
    y_pred = sigmoid(torch.mm(x, theta))
    return y_pred

def torch_process_logistic(args):
    load_dotenv()
    os.getenv("AWS_ACCESS_KEY_ID")
    os.getenv("AWS_SECRET_ACCESS_KEY")
    os.getenv("MLFLOW_S3_ENDPOINT_URL")
    log_folder = args["log_folder"]
    output_dir = args["output_dir"]
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
        bow_word_frequency = joblib.load(open(log_folder + '/bow_word_frequency.pkl', 'rb'))
        train_X_tweet = joblib.load(open(log_folder + '/train_X_tweet.pkl', 'rb'))
        test_X_tweet = joblib.load(open(log_folder + '/test_X_tweet.pkl', 'rb'))
        train_Y = joblib.load(open(log_folder + '/train_Y.pkl', 'rb'))
        test_Y = joblib.load(open(log_folder + '/test_Y.pkl', 'rb'))
        train_X_Tensor = torch.zeros((len(train_X_tweet), 3))
        for index, tweet in enumerate(train_X_tweet):
            train_X_Tensor[index, :] = extract_features(tweet, bow_word_frequency)

        test_X_Tensor = torch.zeros((len(test_X_tweet), 3))
        for index, tweet in enumerate(test_X_tweet):
            test_X_Tensor[index, :] = extract_features(tweet, bow_word_frequency)
        print(f"train_X_Tensor {train_X_Tensor.shape}, test_X_Tensor {test_X_Tensor.shape}")
        type(train_X_Tensor)
        torch.manual_seed(1)
        J, theta = gradientDescent(train_X_Tensor,
                                   torch.reshape(torch.Tensor(train_Y.to_numpy()), (-1, 1)),
                                   torch.zeros((3, 1)), 1e-7, 1000, 0.1)
        print(f"The cost after training is {J:.8f}.")
        predicted_probs = predict_tweet(test_X_Tensor, theta)
        prediceted_probs = torch.tensor(predicted_probs)
        predicted_labels = torch.where(predicted_probs > 0.5, torch.tensor(1), torch.tensor(0))
        print(
            f"Pytorch of logistic regression accuracy is {len(predicted_labels[predicted_labels == torch.reshape(torch.Tensor(test_Y.to_numpy()), (-1, 1))]) / len(test_Y) * 100:.2f}")

        if not os.path.isdir(log_folder + '/torch'):
            os.makedirs(log_folder + '/torch')
        torch_folder = log_folder + '/torch'
        joblib.dump(theta, torch_folder + '/torch.pkl')

        torch_score = len(
            predicted_labels[predicted_labels == torch.reshape(torch.Tensor(test_Y.to_numpy()), (-1, 1))]) / len(test_Y)
        metric_dict = {}
        metric_dict["torch_score"] = torch_score
        mlflow.log_metrics(metric_dict)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        files = os.listdir(log_folder)
        print("FILES:", files)
        shutil.copytree(torch_folder, output_dir, dirs_exist_ok=True)
        file1 = open(os.path.join(output_dir, "torchs_score.txt"), "w")
        file1.write(str(torch_score))
        file1.close()
        mlflow.log_artifacts(output_dir, "torch_output")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--tracking_uri", type=str, default="http://127.0.0.1:5001")
    parser.add_argument("--experiment_id", type=str, default=1)
    parser.add_argument("--run_id", type= str, default="")
    args = parser.parse_args()
    return args
if __name__ =="__main__":
    args = parse_args()
    args = vars(args)
    torch_process_logistic(args)