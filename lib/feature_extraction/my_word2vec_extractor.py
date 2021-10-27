import os
import time

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
from config import CODE_DIR
import string

TRANSLATE = str.maketrans({x: ' ' for x in string.punctuation})


class CBOWEncoder(object):
    _backup = None
    name = 'CBOW'

    def __init__(self):
        if CBOWEncoder._backup is None:
            self.model = None
        else:
            self.model = CBOWEncoder._backup

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
            model_path = os.path.join(CODE_DIR, "models/word2vec_models/my_w2v.txt")
            print("loading cbow encoder ...")
            start_time = time.time()
            self.model = KeyedVectors.load_word2vec_format(model_path)
            print("cbow loading time: " + str(time.time() - start_time))
            CBOWEncoder._backup = self.model
        features = [self.get_sentence_representation(sent) for sent in sentences]
        return normalize(np.array(features), norm='l1', axis=1)

    @staticmethod
    def clear():
        del CBOWEncoder._backup
        CBOWEncoder._backup = None

    def __str__(self):
        return self.name
