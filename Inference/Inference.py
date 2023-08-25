import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
import pickle
import os
import logging
log = logging.getLogger()
class Inference(object):
    def __init__(self):
        self.model_loaded = False
        self.tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        self.stopwords_en = stopwords.words('english')
        self.punctuation_en = string.punctuation
        self.stemmer = PorterStemmer()
        self.sklearn_folder= "sklearn_folder"
        self.emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
                  ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
                  ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed',
                  ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
                  '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
                  '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
                  ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

        ## Defining set containing all stopwords in english.
        self.stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                        'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
                        'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
                        'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
                        'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                        'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                        'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                        'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
                        'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're',
                        's', 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such',
                        't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                        'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                        'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
                        'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
                        'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
                        "youve", 'your', 'yours', 'yourself', 'yourselves']

    def preprocess(self, textdata):
        processedText = []

        # Create Lemmatizer and Stemmer.
        wordLemm = WordNetLemmatizer()

        # Defining regex patterns.
        urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
        userPattern = '@[^\s]+'
        alphaPattern = "[^a-zA-Z0-9]"
        sequencePattern = r"(.)\1\1+"
        seqReplacePattern = r"\1\1"

        for tweet in textdata:
            tweet = tweet.lower()

            # Replace all URls with 'URL'
            tweet = re.sub(urlPattern, ' URL', tweet)
            # Replace all emojis.
            for emoji in self.emojis.keys():
                tweet = tweet.replace(emoji, "EMOJI" + self.emojis[emoji])
                # Replace @USERNAME to 'USER'.
            tweet = re.sub(userPattern, ' USER', tweet)
            # Replace all non alphabets.
            tweet = re.sub(alphaPattern, " ", tweet)
            # Replace 3 or more consecutive letters by 2 letter.
            tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
            tweetwords = ''
            for word in tweet.split():
                # Checking if the word is a stopword.
                # if word not in stopwordlist:
                if len(word) > 1:
                    # Lemmatizing the word.
                    word = wordLemm.lemmatize(word)
                    tweetwords += (word + ' ')

            processedText.append(tweetwords)
        return processedText

    def load_models(self):
        '''
        Replace '..path/' by the path of the saved models.
        '''
        # Load the vectoriser.
        file = open(os.path.join(self.sklearn_folder,'vectoriser-ngram-(1,2).pickle'), 'rb')
        vectoriser = pickle.load(file)
        file.close()
        # Load the LR Model.
        file = open(os.path.join(self.sklearn_folder,'Sentiment-LR.pickle'), 'rb')
        LRmodel = pickle.load(file)
        file.close()
        return vectoriser, LRmodel

    def predict(self,  X, features_names=None):
        if self.model_loaded== False:
            print("LOADING MODEL")
            self.vectorizer, self.model = self.load_models()
            self.model_loaded=True
        # Predict the sentiment
        log.info("X:", X)
        textdata = self.vectorizer.transform(self.preprocess(X))
        log.info("Transformed Data")
        sentiment = self.model.predict(textdata)
        # Make a list of text with sentiment.
        log.info("Got Predictions")
        data = {}
        emotion_dict= {0: "Sad", 1: "Happy"}
        for text, pred in zip(X, sentiment):
            data[text]= emotion_dict[pred]
        log.info("Sending Predictions back!!")
        return data


