#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Michael Ruppert <michael.ruppert@fau.de>

import json
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from skmultiflow.trees import LabelCombinationHoeffdingTreeClassifier, HoeffdingTreeClassifier
from skmultiflow.bayes import NaiveBayes
from skmultiflow.lazy import KNNClassifier
from skmultiflow.meta import ClassifierChain
import numpy as np
from PyQt6 import QtWidgets, uic, QtGui

BASE_FILE = "tweets.ndjson"


class LabelPredict:
    def __init__(self, texts: list):
        self.tokenizer = TfidfVectorizer()
        self.tokenizer.fit(texts)

        self.labels_sent = {"POSITIVE": np.array([1, 0, 0]), "NEUTRAL": np.array([0, 1, 0]),
                            "NEGATIVE": np.array([0, 0, 1])}
        self.labels_sent = {"POSITIVE": 0, "NEUTRAL": 1,
                            "NEGATIVE": 2}
        self.reverse_sent = {0: {"POSITIVE": True, "NEUTRAL": False,
                                 "NEGATIVE": False},
                             1: {"POSITIVE": False, "NEUTRAL": True,
                                 "NEGATIVE": False},
                             2: {"POSITIVE": False, "NEUTRAL": False,
                                 "NEGATIVE": True}}

        self.labels_relevance = ["Irrelevant"]
        self.labels = []
        self.lcc = ClassifierChain(SGDClassifier(max_iter=100, loss='log', random_state=1))
        self.clrel = KNNClassifier()
        self.clsent = KNNClassifier()

    def _labels2array(self, labeldict: dict):
        target = []
        for label in self.labels:
            if label in labeldict and labeldict[label] == True:
                target.append(1)
            else:
                target.append(0)
        return np.array(target)

    def retrain(self, labeled_tweets: list):
        labels = set()
        for tweet in labeled_tweets:
            if "labels" in tweet and len(tweet["labels"]) > 0:
                labels.update([l for l in tweet["labels"] if not (l in self.labels_sent or l in self.labels_relevance)])
        self.labels = list(labels)
        assert "Irrelevant" not in self.labels, "Something went wrong"
        self.lcc = ClassifierChain(SGDClassifier(max_iter=100, loss='log', random_state=1))
        self.clrel = KNNClassifier()
        self.clsent = KNNClassifier()
        X, y, ys, yr = [], [], [], []
        for tweet in labeled_tweets:
            if "labels" in tweet and len(tweet["labels"]) > 0:
                X.append(tweet["tweet"])
                y.append(self._labels2array(tweet["labels"]))
                sls = [l for l, v in tweet["labels"].items() if l in self.labels_sent and v]
                if len(sls) == 1:
                    ys.append(self.labels_sent[sls[0]])
                else:
                    ys.append(self.labels_sent["NEUTRAL"])
                if self.labels_relevance[0] in tweet["labels"] and tweet["labels"][self.labels_relevance[0]]:
                    yr.append(1)
                else:
                    yr.append(0)

        X = np.array(self.tokenizer.transform(X).todense())
        y = np.array(y)
        ys = np.array(ys)
        yr = np.array(yr)
        self.clsent.fit(X, ys)
        print("Trained Sentiment Classifier")
        self.clrel.fit(X, yr)
        print("Trained Relevance Classifier")
        X2, y2 = [], []
        for Xe, ye in zip(X, y):
            if ye.sum() > 0:
                X2.append(Xe)
                y2.append(ye)
        X = np.array(X2)
        y = np.array(y2)
        self.lcc.fit(X, y)
        print("Trained Catecorical Classifier")

    def predict(self, text: str):
        X = np.array(self.tokenizer.transform([text]).todense()).reshape((1, -1))
        predicted = self.lcc.predict(X)
        labels_add = {label: bool(value) for label, value in zip(self.labels, predicted.flatten())}
        sent_pred = self.clsent.predict(X)
        labels_add.update(self.reverse_sent[sent_pred.flatten()[0]])

        assert "POSITIVE" in labels_add, "Klassifikation nicht eindeutig"

        if self.clrel.predict(X) == np.array([1]):
            labels_add[self.labels_relevance[0]] = True
        else:
            labels_add[self.labels_relevance[0]] = False
        return labels_add

    def train_item(self, tweet):
        text = tweet["tweet"]
        labeldict = tweet["labels"]
        for l in labeldict:
            if l not in self.labels and l not in self.labels_relevance and l not in self.labels_sent:
                print("RETRAIN!")
                return False
        y = self._labels2array(labeldict).reshape((1, -1))
        X = np.array(self.tokenizer.transform([text]).todense()).reshape((1, -1))

        sls = [l for l, v in labeldict.items() if l in self.labels_sent and v]
        if len(sls) == 1:
            ys = self.labels_sent[sls[0]]
        else:
            ys = self.labels_sent["NEUTRAL"]
        ys = np.array([ys])
        if self.labels_relevance[0] in labeldict and labeldict[self.labels_relevance[0]]:
            yr = np.array([1])
        else:
            yr = np.array([0])
        if y.sum() > 0:
            self.lcc.partial_fit(X, y)
        if yr.sum() > 0:
            self.clrel.partial_fit(X, yr)
        if ys.sum() > 0:
            self.clsent.partial_fit(X, ys)
        return True


class Boxes(QtWidgets.QWidget):
    def _add_qbox(self):
        if len(self.vboxes) * 4 < len(self.boxes):
            p = QtWidgets.QWidget()
            k = QtWidgets.QVBoxLayout(p)
            self.vboxes.append(k)
            self.layout.addWidget(p)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.layout = parent.findChild(QtWidgets.QHBoxLayout, "boxes")
        self.vboxes = []
        self.boxes = []

        self.setAutoFillBackground(True)

        for tag in ["Irrelevant", "NEGATIVE", "POSITIVE", "NEUTRAL"]:
            self.boxes.append(QtWidgets.QCheckBox(self))
            self.boxes[-1].setText(tag)
            self.boxes[-1].setChecked(False)
            self.boxes[-1].setMinimumWidth(80)
            self.boxes[-1].adjustSize()
            self.boxes[-1].setAutoFillBackground(True)
            self.boxes[-1].setMinimumHeight(36)

            self._add_qbox()
            self.vboxes[-1].addWidget(self.boxes[-1])

    def add_label(self):
        existing = {box.text(): box for box in self.boxes}
        text = self.parent().new_label.text()
        self.parent().new_label.clear()
        if len(text) < 1:
            return
        if text in existing:
            existing[text].setChecked(True)
            return

        self.boxes.append(QtWidgets.QCheckBox(self))
        self.boxes[-1].setText(text)
        self.boxes[-1].setChecked(True)
        self.boxes[-1].setMinimumWidth(80)
        self.boxes[-1].adjustSize()
        self.boxes[-1].setAutoFillBackground(True)
        self.boxes[-1].setMinimumHeight(36)

        self._add_qbox()

        self.vboxes[-1].addWidget(self.boxes[-1])

    def add_labels(self, labels: set):
        existing = {box.text() for box in self.boxes}
        for label in labels - existing:
            self.boxes.append(QtWidgets.QCheckBox(self))
            self.boxes[-1].setText(label)
            self.boxes[-1].setMinimumWidth(80)
            self.boxes[-1].adjustSize()
            self.boxes[-1].setAutoFillBackground(True)
            self.boxes[-1].setMinimumHeight(36)
            self._add_qbox()
            self.vboxes[-1].addWidget(self.boxes[-1])

    def clear(self):
        for box in self.boxes:
            box.setChecked(False)

    def get_checked(self):
        return {box.text(): box.isChecked() for box in self.boxes}

    def check(self, label, tval):
        for box in self.boxes:
            if label == box.text():
                box.setChecked(tval)


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('annotator.ui', self)
        self.setWindowIcon(QtGui.QIcon("pen.svg"))
        self.setAccessibleName("qAnnotator")
        self.setWindowTitle("qAnnotator")

        self.boxes = Boxes(parent=self)

        self.new_label = self.findChild(QtWidgets.QLineEdit, "textEdit")
        self.new_label.returnPressed.connect(self.boxes.add_label)
        self.tweets = {}
        self.tweets_processed = []
        self.read_tweets()
        self.tweets_not_yet_processed = [id for id in self.tweets if id not in self.tweets_processed]
        self.current_tweet = None
        self.labelpredict = LabelPredict([tweet["tweet"] for tweet in self.tweets.values()])
        if len(self.tweets_processed) > 0:
            self.labelpredict.retrain([self.tweets[t] for t in self.tweets_processed])

        self.text = self.findChild(QtWidgets.QTextBrowser, "tweet_text")

        self.progress = self.findChild(QtWidgets.QProgressBar, "progress")
        self.progress.setMaximum(len(self.tweets))
        self.progress.setValue(len(self.tweets_processed))

        self.load_tweet(self.tweets_not_yet_processed.pop(0))

        self.next = self.findChild(QtWidgets.QPushButton, "button_next")
        self.next.clicked.connect(self.next_tweet)
        self.last = self.findChild(QtWidgets.QPushButton, "button_last")
        self.last.clicked.connect(self.last_tweet)

        self.show()

    def next_tweet(self):
        tweet = self.current_tweet
        new_boxes = self.boxes.get_checked()
        if tweet["labels"] != new_boxes:
            tweet["labels"] = new_boxes
            self.add_tweet(tweet)
        self.tweets_processed.append(tweet["id"])
        if not self.labelpredict.train_item(tweet=tweet):  # this does training
            self.labelpredict.retrain([self.tweets[t] for t in self.tweets_processed])

        if len(self.tweets_not_yet_processed) == 0:
            mb = QtWidgets.QMessageBox()
            mb.setText("All tweets are processed")
            mb.exec()
        else:
            self.load_tweet(self.tweets_not_yet_processed.pop(0))

    def last_tweet(self):
        tweet = self.current_tweet
        new_boxes = self.boxes.get_checked()
        if tweet["labels"] != new_boxes:
            tweet["labels"] = new_boxes
            self.add_tweet(tweet)
        self.tweets_not_yet_processed.insert(tweet["id"], 0)

        if len(self.tweets_processed) == 0:
            mb = QtWidgets.QMessageBox()
            mb.setText("You already see the first Tweet.")
            mb.exec()
        else:
            self.load_tweet(self.tweets_processed.pop(-1))

    def load_tweet(self, id: str):
        self.progress.setValue(len(self.tweets_processed))
        tweet = self.tweets[id]
        self.current_tweet = tweet
        if "labels" not in tweet:
            tweet["labels"] = {}
        self.boxes.clear()
        if len(tweet["labels"]) < 1:
            for label, true in self.labelpredict.predict(tweet["tweet"]).items():
                self.boxes.check(label, true)
        else:
            for label, true in tweet["labels"].items():
                self.boxes.check(label, true)
        text = tweet["tweet"]
        self.text.clear()
        self.text.insertHtml(text)

    def read_tweets(self):
        labels = set()
        for line in open(BASE_FILE, encoding="utf8"):
            tweet = json.loads(line)
            self.tweets[tweet["id"]] = tweet
        for tid, tweet in self.tweets.items():
            if "labels" in tweet and len(tweet["labels"]) > 0:
                self.tweets_processed.append(tid)
                labels.update([l for l in tweet["labels"]])
        self.boxes.add_labels(labels)

    def add_tweet(self, tweet: dict):
        with open(BASE_FILE, "a", encoding="utf8") as f:
            f.write(json.dumps(tweet) + "\n")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec()
