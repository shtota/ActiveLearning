import os
import time

import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
from config import CODE_DIR
import string
from gensim.models import fasttext
from sklearn.feature_extraction.text import CountVectorizer

TRANSLATE = str.maketrans({x: ' ' for x in string.punctuation})


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances or 'clean' in kwargs:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class _Embedding:
    def __init__(self, *args, **kwargs):
        self.model = None
        self.norm = kwargs.get('norm', 'l2')

    def _load_model(self):
        if self.model_path:
            print("loading {} encoder ...".format(self.name))
            start_time = time.time()
            self.model = KeyedVectors.load_word2vec_format(self.model_path, binary=self.model_path.endswith('bin'))
            print("{} loading time: {}".format(self.name, time.time() - start_time))

    def _get_sentence_representation(self, sent):
        sent = sent.translate(TRANSLATE)
        tokens = [x.lower() for x in sent.split(' ') if len(x)]
        tokens = [x for x in tokens if x in self.model.key_to_index]
        if len(tokens):
            return sum([self.model[x] for x in tokens])/len(tokens)
        else:
            print(sent)
            return np.zeros(300)

    def _get_corpus_features(self, sentences):
        return np.array([self._get_sentence_representation(sent) for sent in sentences])

    def prepare_features(self, sentences):
        if self.model is None:
            self._load_model()
        features = self._get_corpus_features(sentences)
        return normalize(features, norm=self.norm, axis=1)

    def __str__(self):
        if self.norm != 'l2':
            return self.name + '_' + self.norm
        return self.name


class CBoW(_Embedding, metaclass=Singleton):
    name = 'cbow'
    model_path = os.path.join(CODE_DIR, "models", "word2vec_models", "my_w2v.txt")


class Glove(_Embedding, metaclass=Singleton):
    name = 'glove'
    model_path = os.path.join(CODE_DIR, "models", 'glove', "glove.6B.300d.txt")


class SkipGram(_Embedding, metaclass=Singleton):
    name = 'skipgram'
    model_path = os.path.join(CODE_DIR, "models", "word2vec_models", "GoogleNews-vectors-negative300.bin")


class FastText(_Embedding, metaclass=Singleton):
    name = 'fasttext'
    model_path = os.path.join(CODE_DIR, "models", "fasttext", "crawl-300d-2M-subword.bin")

    def _load_model(self):
        print("loading fasttext encoder ...")
        start_time = time.time()
        self.model = fasttext.load_facebook_model(self.model_path)
        print("Fasttext loading time: " + str(time.time() - start_time))

    def _get_sentence_representation(self, sent):
        sent = sent.translate(TRANSLATE)
        tokens = [x for x in sent.split(' ') if len(x)]
        return sum([self.model.wv[x] for x in tokens])/len(tokens)


class BoW(_Embedding, metaclass=Singleton):
    name = 'bow'

    def __init__(self, *args, **kwargs):
        self.model_path = ''
        self.vectorizer = CountVectorizer(
            analyzer='word',
            lowercase=True,
            stop_words='english'
        )
        super(BoW, self).__init__(*args, **kwargs)

    def _get_corpus_features(self, sentences):
        return self.vectorizer.fit_transform(sentences)


class Transformer(_Embedding, metaclass=Singleton):
    name = 'transformer'

    def _get_corpus_features(self, sentences):
        embeddings = self.model.encode([' '.join([x.lower() for x in s.translate(TRANSLATE).split(' ') if len(x)])
                                        for s in sentences], convert_to_numpy=True)
        return embeddings

    def _load_model(self):
        start_time = time.time()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('paraphrase-MiniLM-L12-v2', device='cuda')
        print("loaded transformer", time.time() - start_time)


ENCODER_NAME_TO_CLASS = {x.name: x for x in [Transformer, Glove, FastText, CBoW, SkipGram, BoW]}
ALL_EMBEDDINGS = sorted(ENCODER_NAME_TO_CLASS.keys())
