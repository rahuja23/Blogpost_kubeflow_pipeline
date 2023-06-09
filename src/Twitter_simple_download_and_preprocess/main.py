import re
import string
import pandas as pd
from random import shuffle
import nltk
import joblib
import argparse
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from nltk import data
from typing import NamedTuple

class Twitter_Preprocess():
    def __init__(self):
        self.tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                                        reduce_len=True)
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

def build_bow_dict(tweets, labels):
    freq = {}
    for tweet, label in list(zip(tweets, labels)):
        for word in tweet:
            freq[(word, label)] = freq.get((word, label), 0) + 1
    return freq
def twitter_sample_download_and_preprocess(args) -> NamedTuple('Outputs', [('logdir', str)]):
    log_folder = args.log_folder
    print(log_folder)
    data.path.append(log_folder)
    data.path.append(log_folder)
    nltk.download('twitter_samples', download_dir=log_folder)
    nltk.download('stopwords', download_dir=log_folder)
    pos_tweets = twitter_samples.strings('positive_tweets.json')
    neg_tweets = twitter_samples.strings('negative_tweets.json')
    print(f"positive sentiment GOOD total samples {len(pos_tweets)}")
    print(f"negative sentiment  Bad total samples {len(neg_tweets)}")
    twitter_text_processor = Twitter_Preprocess()
    processed_pos_tweets = twitter_text_processor.preprocess(pos_tweets)
    processed_neg_tweets = twitter_text_processor.preprocess(neg_tweets)

    labels = [1 for i in range(len(processed_pos_tweets))]
    labels.extend([0 for i in range(len(processed_neg_tweets))])

    twitter_processed_corpus = processed_pos_tweets + processed_neg_tweets
    bow_word_frequency = build_bow_dict(twitter_processed_corpus, labels)

    shuffle(processed_pos_tweets)
    shuffle(processed_neg_tweets)

    positive_tweet_label = [1 for i in processed_pos_tweets]
    negative_tweet_label = [0 for i in processed_neg_tweets]

    tweet_df = pd.DataFrame(list(zip(twitter_processed_corpus,
                                     positive_tweet_label + negative_tweet_label)),
                            columns=["processed_tweet", "label"])

    train_X_tweet, test_X_tweet, train_Y, test_Y = train_test_split(tweet_df["processed_tweet"],
                                                                    tweet_df["label"],
                                                                    test_size=0.20,
                                                                    stratify=tweet_df["label"])

    print(f"train_X_tweet {train_X_tweet.shape}, test_X_tweet {test_X_tweet.shape}")
    print(f"train_Y {train_Y.shape}, test_Y {test_Y.shape}")
    joblib.dump(bow_word_frequency, log_folder + '/bow_word_frequency.pkl')
    joblib.dump(train_X_tweet, log_folder + '/train_X_tweet.pkl')
    joblib.dump(test_X_tweet, log_folder + '/test_X_tweet.pkl')
    joblib.dump(train_Y, log_folder + '/train_Y.pkl')
    joblib.dump(test_Y, log_folder + '/test_Y.pkl')
    return log_folder
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", type=str, default="")
    args = parser.parse_args()
    return args
if __name__ =="__main__":
    args = parse_args()
    folder = twitter_sample_download_and_preprocess(args)


