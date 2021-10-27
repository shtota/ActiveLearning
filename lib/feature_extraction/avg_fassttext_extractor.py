import os
import time
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from gensim.models import fasttext
from config import CODE_DIR
import string

TRANSLATE = str.maketrans({x: ' ' for x in string.punctuation})


class FastTextEncoder(object):
    _backup = None
    name = 'fasttext'

    def __init__(self):
        if FastTextEncoder._backup is None:
            self.model = None
        else:
            self.model = FastTextEncoder._backup

    def get_sentence_representation(self, sent):
        sent = sent.translate(TRANSLATE)
        tokens = [x for x in sent.split(' ') if len(x)]
        return sum([self.model.wv[x] for x in tokens])/len(tokens)

    def prepare_features(self, sentences):
        if self.model is None:
            model_path = os.path.join(CODE_DIR, "models/fasttext/crawl-300d-2M-subword.bin")
            print("loading fasttext encoder ...")
            start_time = time.time()
            self.model = fasttext.load_facebook_model(model_path)
            print("Fasttext loading time: " + str(time.time() - start_time))
            FastTextEncoder._backup = self.model
        features = [self.get_sentence_representation(sent) for sent in sentences]
        return normalize(np.array(features), norm='l1', axis=1)

    @staticmethod
    def clear():
        del FastTextEncoder._backup
        FastTextEncoder._backup = None

    def __str__(self):
        return self.name
