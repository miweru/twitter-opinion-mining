from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from tensorflow import keras
import typing
import numpy as np
from embeddings import CharEmbedding, BERTweet, LDAembedding, BERTweetCONV, ELECTRA


class TweetLabelPredictor:
    def __init__(self, trainpath="./trained_on_all_large/"):
        """


        """
        self.char, self.bert, self.lda, self.bert_conv, self.electra = CharEmbedding(), BERTweet(), LDAembedding(), BERTweetCONV(), ELECTRA()
        self.model = keras.models.load_model(filepath=trainpath, compile=True)

    def enrich_datast(self, tweets: typing.List[typing.Dict], cutoff=0.4):
        output = self.get_prediction_data([tweet["tweet"] for tweet in tweets])
        labels = []
        for polarity, relevance, spam, joke, political, technology, problem, money, music, brand in zip(*output):
            s_pols = {"POSITIVE": False, "NEUTRAL": False, "NEGATIVE": False}
            s_pols[("POSITIVE", "NEUTRAL", "NEGATIVE")[polarity.argmax()]] = True
            s_pols["Irrelevant"] = True if relevance.argmax() == 1 else False
            s_pols["spam"] = True if spam[1] > cutoff else False
            s_pols["joke"] = True if joke[1] > cutoff else False
            s_pols["political"] = True if political[1] > cutoff else False
            s_pols["technology"] = True if technology[1] > cutoff else False
            s_pols["problem"] = True if problem[1] > cutoff else False
            s_pols["money"] = True if money[1] > cutoff else False
            s_pols["music"] = True if music[1] > cutoff else False
            s_pols["brand"] = True if brand[1] > cutoff else False
            labels.append(s_pols)
        for tweet, label in zip(tweets, labels):
            tweet["labels"] = label
        return tweets

    def _get_labels(self):
        for tweet in self._tweets:
            yield tweet["labels"]

    def _get_hashtags(self):
        for tweet in self._tweets:
            yield tweet["hashtags"]

    def get_prediction_data(self, tweet_texts: typing.Iterable[typing.Text]) -> typing.Dict[str, np.array]:
        """

        :return:
        """
        return self.model.predict(
            [self.char(tweet_texts), self.bert(tweet_texts), self.bert_conv(tweet_texts), self.electra(tweet_texts),
             self.lda(tweet_texts)])
