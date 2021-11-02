import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from config import CODE_DIR
import string

TRANSLATE = str.maketrans({x: ' ' for x in string.punctuation})


class GloveEncoder(object):
    _backup = None
    name = 'glove'

    def __init__(self):
        if GloveEncoder._backup is None:
            self.glove_model = None
            self.vocab_set = None
        else:
            self.glove_model = GloveEncoder._backup
            self.vocab_set = set(self.glove_model.dictionary.keys())

    def all_in_vocab(self, sent):
        self.initialize_vocab()
        # import lib.Constants as cn
        # cn.add_experiment_param('glove_token_split')
        # return " ".join(filter(lambda word: word in vocab_set, pos_tagger.tokenize_sent(sent)))
        return " ".join([word for word in sent.lower().split() if word in self.vocab_set])

    def initialize_vocab(self):
        if self.vocab_set is None:
            self.vocab_set = set(self.glove_model.dictionary.keys())

    def get_sentence_representation(self, sent):
        sent = sent.translate(TRANSLATE)
        reduced_sent = self.all_in_vocab(sent).split()
        if len(reduced_sent) == 0:
            # print "reduced sent: " + str(sent)
            return [0.0] * len(self.glove_model.word_vectors[0])  # zeros representation
        return sum([self.glove_model.word_vectors[self.glove_model.dictionary[word]].__array__() for word in reduced_sent])/len(reduced_sent)

    def prepare_features(self, sentences):
        # type: (pd.DataFrame) -> np.array
        """
        extracts features for the sentences in $df
        :param df: a DataFrame object with the columns $col_names
        :param col_names: namedtuple containing all column names
        :return: a DataFrame object with all the sentences with their feature representation
        """
        if self.glove_model is None:
            from glove import Glove
            model_path = os.path.join(CODE_DIR, "models", 'glove', "glove.6B.300d.txt")
            print("loading glove encoder ...")
            start_time = time.time()
            self.glove_model = Glove.load_stanford(model_path)
            print("AvgGloveExtractor loading time: " + str(time.time() - start_time))
            GloveEncoder._backup = self.glove_model
        features = [self.get_sentence_representation(sent) for sent in sentences]
        return normalize(np.array(features), norm='l2', axis=1)

    @staticmethod
    def clear():
        del GloveEncoder._backup
        GloveEncoder._backup = None
    def __str__(self):
        return self.name
