#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Michael Ruppert <michael.ruppert@fau.de>

import json
import pathlib
import re
import typing
from collections import Counter

import numpy as np
from emoji import demojize
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm
from transformers import AutoTokenizer, TFAutoModel


class TweetNormalisation:
    """
    Geklaut von https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py
    """

    def __init__(self):
        self.tokenizer = TweetTokenizer()

    @staticmethod
    def normalizeToken(token):
        lowercased_token = token.lower()
        if token.startswith("@"):
            return "@USER"
        elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
            return "HTTPURL"
        elif len(token) == 1:
            return demojize(token)
        else:
            if token == "’":
                return "'"
            elif token == "…":
                return "..."
            else:
                return token

    def normalizeTweet(self, tweet):
        tokens = self.tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
        normTweet = " ".join([self.normalizeToken(token) for token in tokens])

        normTweet = normTweet.replace("cannot ", "can not ").replace("n't ", " n't ").replace("n 't ", " n't ").replace(
            "ca n't", "can't").replace("ai n't", "ain't")
        normTweet = normTweet.replace("'m ", " 'm ").replace("'re ", " 're ").replace("'s ", " 's ").replace("'ll ",
                                                                                                             " 'll ").replace(
            "'d ", " 'd ").replace("'ve ", " 've ")
        normTweet = normTweet.replace(" p . m .", "  p.m.").replace(" p . m ", " p.m ").replace(" a . m .",
                                                                                                " a.m.").replace(
            " a . m ", " a.m ")

        normTweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet)
        normTweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet)
        normTweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet)

        return " ".join(normTweet.split())

    def __call__(self, text):
        return self.normalizeTweet(text)


class InputEmbedding:
    def __init__(self, workdir="./embedding-models", name=""):
        """


        """
        self._workdir = pathlib.Path(workdir).expanduser().resolve()
        if not self._workdir.exists():
            self._workdir.mkdir()
        self._name = name
        self._load()

    def _load(self):
        """

        :return:
        """

    def pretrain(self, texts: typing.Iterable[typing.Text]):
        """

        :param texts:
        :return:
        """

    def get_train_data(self, texts: typing.Iterable[typing.Text]) -> np.array:
        """

        :param texts:
        :return:
        """

    def __call__(self, texts: typing.Iterable[typing.Text]) -> np.array:
        return self.get_train_data(texts=texts)


class CharEmbedding(InputEmbedding):
    def __init__(self, workdir="./embedding-models", name="char-embedding"):
        """
        Erstellt durch Aufruf von Pretrain ein Vokabular
        :param workdir:
        :param name:
        """
        super(CharEmbedding, self).__init__(workdir=workdir, name=name)
        self.min_count = 500
        self.seq_len = 240

    def _load(self):
        modeldir = self._workdir.joinpath("model_{}.json".format(self._name))
        if not modeldir.exists():
            return False
        with open(modeldir, encoding="utf8") as f:
            self.c2v = json.load(f)

    def pretrain(self, texts: typing.Iterable[typing.Text]):
        chars = Counter()
        for text in texts:
            assert isinstance(text, str)
            chars.update(list(text))
        self.c2v = {c: i for i, (c, h) in enumerate(chars.most_common()) if h > self.min_count}
        modeldir = self._workdir.joinpath("model_{}.json".format(self._name))
        with open(modeldir, "w", encoding="utf8") as f:
            json.dump(self.c2v, fp=f, ensure_ascii=False)

    def get_train_data(self, texts: typing.Iterable[typing.Text]) -> np.array:
        batch = []
        for text in texts:
            text = [c for c in text if c in self.c2v]
            text_a = [np.zeros(len(self.c2v), dtype=np.bool) for _ in range(self.seq_len - len(text))]
            for c in text:
                c_a = np.zeros(len(self.c2v), dtype=np.bool)
                c_a[self.c2v[c]] = True
                text_a.append(c_a)
            batch.append(np.array(text_a[:self.seq_len]))
        return np.stack(batch)


class BERTweet(InputEmbedding):
    def _load(self):
        """

        :return:
        """
        self._tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
        self._model = TFAutoModel.from_pretrained("vinai/bertweet-base")
        self._normalizer = TweetNormalisation()
        self.OUTPUT = "pooler_output"

    def _get_train_data(self, texts: typing.Iterable[typing.Text]) -> np.array:
        """

        :param texts:
        :return:
        """
        inputs = self._tokenizer([self._normalizer(t) for t in texts], return_tensors="tf", padding=True)
        outputs = self._model(**inputs)
        return outputs[self.OUTPUT].numpy()

    def get_train_data(self, texts: typing.Iterable[typing.Text]) -> np.array:
        batches = [[]]
        for text in texts:
            if len(batches[-1]) >= 256:
                batches.append([])
            batches[-1].append(text)
        batches = [self._get_train_data(batch) for batch in tqdm(batches)]
        return np.vstack(batches)


class BERTweetCONV(BERTweet):
    def _load(self):
        """

        :return:
        """
        self._tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
        self._model = TFAutoModel.from_pretrained("vinai/bertweet-base")
        self._normalizer = TweetNormalisation()
        self.OUTPUT = "last_hidden_state"

    def _get_train_data(self, texts: typing.Iterable[typing.Text]) -> np.array:
        """

        :param texts:
        :return:
        """
        inputs = self._tokenizer([self._normalizer(t) for t in texts], return_tensors="tf", padding="max_length",
                                 max_length=128)
        outputs = self._model(**inputs)
        return outputs[self.OUTPUT].numpy()


class ELECTRA(BERTweet):
    def _load(self):
        """

        :return:
        """
        self._tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator", use_fast=False)
        self._model = TFAutoModel.from_pretrained("google/electra-base-discriminator")
        self._normalizer = TweetNormalisation()
        self.OUTPUT = 'last_hidden_state'

    def _get_train_data(self, texts: typing.Iterable[typing.Text]) -> np.array:
        """

        :param texts:
        :return:
        """
        inputs = self._tokenizer([self._normalizer(t) for t in texts], return_tensors="tf", padding="max_length",
                                 max_length=128, truncation=True)
        outputs = self._model(**inputs)
        return outputs[self.OUTPUT].numpy()


class LDAembedding(InputEmbedding):
    def __init__(self, workdir="./embedding-models", name="lda-embedding"):
        """
        Erstellt durch Aufruf von Pretrain ein Vokabular
        :param workdir:
        :param name:
        """
        super(LDAembedding, self).__init__(workdir=workdir, name=name)
        self._normalizer = TweetNormalisation()

    def _load(self):
        modeldir = self._workdir.joinpath("ldamodel_{}".format(self._name))
        if not modeldir.exists():
            return False
        self._lda = LdaMulticore.load(str(modeldir))
        self._dictionary = Dictionary.load(str(self._workdir.joinpath("dictionary_{}.gz".format(self._name))))

    def pretrain(self, texts: typing.Iterable[typing.Text]):
        texts = [self._normalizer(text).split() for text in tqdm(texts)]
        self._dictionary = Dictionary(texts, prune_at=200000)
        corpus = [self._dictionary.doc2bow(text) for text in tqdm(texts)]
        self._lda = LdaMulticore(corpus=corpus, id2word=self._dictionary, workers=15, num_topics=50)

        self._dictionary.save(str(self._workdir.joinpath("dictionary_{}.gz".format(self._name))))
        self._lda.save(str(self._workdir.joinpath("ldamodel_{}".format(self._name))))

    def get_train_data(self, texts: typing.Iterable[typing.Text]) -> np.array:
        to_array = lambda x: np.array([v for _, v in self._lda.get_document_topics(x, minimum_probability=0)])
        return np.stack([to_array(self._dictionary.doc2bow(self._normalizer(text).split())) for text in texts])
