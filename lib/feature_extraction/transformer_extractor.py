import os
import time

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
from config import CODE_DIR
import string
from sentence_transformers import SentenceTransformer

TRANSLATE = str.maketrans({x: ' ' for x in string.punctuation})


class TransformerEncoder(object):
    _backup = None
    name = 'Transformer'

    def __init__(self):
        self.model = None

    def get_sentence_representation(self, sent):
        sent = sent.translate(TRANSLATE)
        tokens = [x.lower() for x in sent.split(' ') if len(x)]
        return ' '.join(tokens)

    def prepare_features(self, sentences):
        start_time = time.time()
        self.model = SentenceTransformer('paraphrase-MiniLM-L12-v2',device='cuda')
        print("loaded transformer", time.time() - start_time)
        embeddings = self.model.encode([self.get_sentence_representation(s) for s in sentences], convert_to_numpy=True)
        print("finished embedding", time.time() - start_time)

        return normalize(embeddings, norm='l1', axis=1)

    @staticmethod
    def clear():
        pass

    def __str__(self):
        return self.name
