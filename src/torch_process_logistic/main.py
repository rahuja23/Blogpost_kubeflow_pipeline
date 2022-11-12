import torch
import joblib
import os
from typing import NamedTuple
import argparse
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

def torch_process_logistic(args) -> NamedTuple('Outputs', [('logdir',str), ('torchdir',str),  ('torchscore',float)]):
    log_folder = args.log_folder
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

    return ([log_folder, torch_folder, torch_score])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", type=str, default="")
    args = parser.parse_args()
    return args
if __name__ =="__main__":
    args = parse_args()
    folder = torch_process_logistic(args)