import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
import time

import numpy as np
from sklearn.preprocessing import normalize
import string

TRANSLATE = str.maketrans({x: ' ' for x in string.punctuation})


class BoWExtractor(object):
    _backup = None
    name = 'bow'

    def __init__(self):
        self.vectorizer = CountVectorizer(
            analyzer='word',
            lowercase=True,
            stop_words='english'
        )

    def prepare_features(self, sentences, fit=True):
        if fit:
            features = self.vectorizer.fit_transform(sentences)
        else:
            features = self.vectorizer.transform(sentences)
        res = normalize(features, norm='l1', axis=1)
        return res

    @staticmethod
    def clear():
        pass

    def __str__(self):
        return self.name
