from .avg_glove_extractor import GloveEncoder
from .avg_fassttext_extractor import FastTextEncoder
from .avg_word2vec_extractor import Word2VecEncoder
from .BOW_extractor import BoWExtractor
from .my_word2vec_extractor import CBOWEncoder
from .transformer_extractor import TransformerEncoder

ENCODER_NAME_TO_CLASS = {'fasttext': FastTextEncoder, 'glove': GloveEncoder, 'word2vec': Word2VecEncoder,
                         'bow': BoWExtractor, 'cbow': CBOWEncoder, 'transformer': TransformerEncoder}

ALL_ENCODERS = sorted(ENCODER_NAME_TO_CLASS.keys())