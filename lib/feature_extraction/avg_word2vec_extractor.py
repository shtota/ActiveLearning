import os
import time

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from config import CODE_DIR
import string
from sklearn.preprocessing import normalize

TRANSLATE = str.maketrans({x: ' ' for x in string.punctuation})


class Word2VecEncoder(object):
    _backup = None
    name = 'word2vec'

    def __init__(self):
        if Word2VecEncoder._backup is None:
           self.model = None
        else:
            self.model = Word2VecEncoder._backup

    def get_sentence_representation(self, sent):
        sent = sent.translate(TRANSLATE)
        tokens = [x.lower() for x in sent.split(' ') if len(x)]
        tokens = [x for x in tokens if x in self.model.key_to_index]
        if len(tokens):
            return sum([self.model[x] for x in tokens])/len(tokens)
        else:
            print(sent)
            return np.zeros(300)

    def prepare_features(self, sentences):
        if self.model is None:
            print("loading w2v encoder ...")
            start_time = time.time()
            self.model = KeyedVectors.load_word2vec_format('models/word2vec_models/GoogleNews-vectors-negative300.bin',
                                                           binary=True)
            print("W2v loading time: " + str(time.time() - start_time))
            Word2VecEncoder._backup = self.model
        features = [self.get_sentence_representation(sent) for sent in sentences]

        return normalize(np.array(features), norm='l1', axis=1)

    @staticmethod
    def clear():
        del Word2VecEncoder._backup
        Word2VecEncoder._backup = None

    def __str__(self):
        return self.name
