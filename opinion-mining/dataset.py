#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Michael Ruppert <michael.ruppert@fau.de>

import typing
import numpy as np
from embeddings import InputEmbedding


class TweetLabelDS:
    def __init__(self, tweets: typing.List[typing.Dict], embeddings: typing.Dict[typing.Text, InputEmbedding]):
        """


        """
        self._input_embeddings = embeddings
        self._tweets = tweets

    def _get_text(self):
        for tweet in self._tweets:
            yield tweet["tweet"]

    def _get_labels(self):
        for tweet in self._tweets:
            yield tweet["labels"]

    def _get_hashtags(self):
        for tweet in self._tweets:
            yield tweet["hashtags"]

    def get_train_data(self) -> typing.Tuple[typing.Dict[str, np.array], typing.Dict[str, np.array]]:
        """

        :return:
        """
        return self.get_X_data(), self.get_y_data()

    def get_X_data(self) -> typing.Dict[str, np.array]:
        """

        :return:
        """
        text = list(self._get_text())
        return {en: embedding(text) for en, embedding in self._input_embeddings.items()}

    def save_train_data(self, name: str):
        fn = "traindata_{}_".format(name)
        np.savez(fn + "Y", **self.get_y_data())
        text = list(self._get_text())
        for en, embedding in self._input_embeddings.items():
            np.savez("{}{}".format(fn, en), embedding(texts=text))

    def get_y_data(self) -> typing.Dict[str, np.array]:
        """

        :return:
        """
        polarity = []
        categories = {n: [] for n in ('spam', 'joke', 'political', 'technology', 'problem', 'money', 'music', 'brand')}
        relevance = []

        for tweet in self._tweets:
            labels = tweet["labels"]
            relevance.append(np.array(True, dtype=np.bool) if labels["Irrelevant"] else np.array(False, dtype=np.bool))
            polarity.append(np.array(0) if labels["POSITIVE"] else np.array(2) if labels["NEGATIVE"] else np.array(1))
            for c, l in categories.items():
                l.append(np.array(True, dtype=np.bool) if labels[c] else np.array(False, dtype=np.bool))
        return dict(polarity=np.array(polarity), relevance=np.array(relevance),
                    **{k: np.array(v) for k, v in categories.items()})

    def unsupervised_embedding_pretrain(self):
        data = list(self._get_text())
        for embedding in self._input_embeddings.values():
            embedding.pretrain(data)
