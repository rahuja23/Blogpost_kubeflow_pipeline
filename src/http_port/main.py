import re
import string
import pandas as pd
from random import shuffle
import torch
import numpy as np
import nltk
import joblib
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from nltk import data
from flask import Flask, render_template, url_for, request
import argparse
class Preprocess():
    def __init__(self):
        self.tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        self.stopwords_en = stopwords.words('english')
        self.punctuation_en = string.punctuation
        self.stemmer = PorterStemmer()

    def __remove_unwanted_characters__(self, tweet):
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
        tweet = re.sub(r'#', '', tweet)
        tweet = re.sub('\S+@\S+', '', tweet)
        tweet = re.sub(r'\d+', '', tweet)
        return tweet

    def __tokenize_tweet__(self, tweet):
        return self.tokenizer.tokenize(tweet)

    def __remove_stopwords__(self, tweet_tokens):
        tweets_clean = []
        for word in tweet_tokens:
            if (word not in self.stopwords_en and
                    word not in self.punctuation_en):
                tweets_clean.append(word)
        return tweets_clean

    def __text_stemming__(self, tweet_tokens):
        tweets_stem = []
        for word in tweet_tokens:
            stem_word = self.stemmer.stem(word)
            tweets_stem.append(stem_word)
        return tweets_stem

    def preprocess(self, tweets):
        tweets_processed = []
        for _, tweet in tqdm(enumerate(tweets)):
            tweet = self.__remove_unwanted_characters__(tweet)
            tweet_tokens = self.__tokenize_tweet__(tweet)
            tweet_clean = self.__remove_stopwords__(tweet_tokens)
            tweet_stems = self.__text_stemming__(tweet_clean)
            tweets_processed.extend([tweet_stems])
        return tweets_processed
def extract_features(processed_tweet, bow_word_frequency):
    features = np.zeros((1,3))
    features[0,0] = 1
    for word in processed_tweet:
        features[0,1] = bow_word_frequency.get((word, 1), 0) + features[0,1]
        features[0,2] = bow_word_frequency.get((word, 0), 0) + features[0,2]
    return features

def sigmoid(z):
    h = 1 / (1+ np.exp(-z))
    return h

def predict_tweet(x, theta_ns):
    y_pred = sigmoid(np.dot(x, theta_ns))
    return y_pred

def extract_features_torch(processed_tweet, bow_word_frequency):
    features = torch.zeros((1,3))
    features[0,0] = 1
    for word in processed_tweet:
        features[0,1] = bow_word_frequency.get((word, 1), 0) + features[0,1]
        features[0,2] = bow_word_frequency.get((word, 0), 0) + features[0,2]
    return features


def sigmoid_torch(z):
    h = 1 / (1 + torch.exp(-z))
    return h


def predict_tweet_torch(x, theta_toc):
    y_pred = sigmoid_torch(torch.mm(x, theta_toc))
    return y_pred

def http_port(args):
    log_folder = args.log_folder
    sklearn_folder = args.sklearn_folder
    logistic_folder = args.logistic_folder
    torch_folder = args.torch_folder
    svm_folder = args.svm_folder
    data.path.append(log_folder)

    app = Flask(__name__, template_folder='/http-port/templates')

    @app.route('/')
    def home():
        return render_template('home.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        text_processor = Preprocess()

        bow_word_frequency = joblib.load(open(log_folder + '/bow_word_frequency.pkl', 'rb'))
        theta_ns = joblib.load(open(logistic_folder + '/logistic.pkl', 'rb'))
        clf = joblib.load(open(sklearn_folder + '/sklearn.pkl', 'rb'))
        theta_toc = joblib.load(open(torch_folder + '/torch.pkl', 'rb'))
        svm = joblib.load(open(svm_folder + '/svm.pkl', 'rb'))

        if request.method == 'POST':
            message = request.form['message']
            data = [message]
            data = text_processor.preprocess(data)

            data_o = str(data)
            data_o = data_o[2:len(data_o) - 2]

            vect = np.zeros((1, 3))
            for index, tweet in enumerate(data):
                vect[index, :] = extract_features(tweet, bow_word_frequency)
            predicted_probs_np = predict_tweet(vect, theta_ns)
            my_prediction_np = np.where(predicted_probs_np > 0.5, 1, 0)

            my_prediction_skl = clf.predict(vect)

            vect_Tensor = torch.zeros((1, 3))
            for index, tweet in enumerate(data):
                vect_Tensor[index, :] = extract_features_torch(
                    tweet, bow_word_frequency)
            predicted_probs_toc = predict_tweet_torch(vect_Tensor, theta_toc)
            my_prediction_toc = torch.where(
                predicted_probs_toc > 0.5, torch.tensor(1), torch.tensor(0))

            my_prediction_svm = svm.predict(vect)

        return render_template('home.html',
                               message=message,
                               data=data_o,
                               my_prediction_np=my_prediction_np,
                               my_prediction_skl=my_prediction_skl,
                               my_prediction_toc=my_prediction_toc,
                               my_prediction_svm=my_prediction_svm)

    if __name__ == '__main__':
        app.run(debug=True, use_reloader=False)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", type=str, default="")
    parser.add_argument("--sklearn_folder", type=str, default="")
    parser.add_argument("--logistic_folder", type=str, default= "")
    parser.add_argument("--torch_folder", type= str, default= "")
    parser.add_argument("--svm_folder", type = str, default="")
    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = parse_args()
    http_port(args)
