
import os
import numpy as np
import pickle
import argparse
from itertools import product
from functools import partial

from libact.base.interfaces import QueryStrategy, ProbabilisticModel
from libact.models import *
from config import CODE_DIR
from feature_extraction import FastTextEncoder, GloveEncoder, BoWExtractor, CBOWEncoder, Word2VecEncoder, TransformerEncoder
from libact.query_strategies import UncertaintySampling, RandomSampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR


class LogisticRegressionStable(ProbabilisticModel):

    def __init__(self, C, *args, **kwargs):
        self.C = C
        self.model = LR(*args, **kwargs)
        self.name = kwargs.pop('name', '')
        self.args = args
        self.kwargs = kwargs

    def train(self, dataset, *args, **kwargs):
        X, y = dataset.format_sklearn()
        self.model = LR(C=self.C/y.shape[0], *self.args, **self.kwargs)
        return self.model.fit(X,y,*args, **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args), **kwargs)

    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue

    def predict_proba(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)


class MaxEvidenceSampling(QueryStrategy):
    def __init__(self, dataset, model, k=100, **kwargs):
        super(MaxEvidenceSampling, self).__init__(dataset, **kwargs)
        self.model = model
        self.k = k
        self.model.train(self.dataset)
        self.train_on_query = kwargs.pop('train_on_query', True)

    def make_query(self):
        dataset = self.dataset
        if self.train_on_query:
            self.model.train(dataset)
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        dvalue = self.model.predict_proba(X_pool)
        scores = np.min(dvalue, axis=1)
        idx = scores.argsort()[-self.k:]
        evidence = np.abs(X_pool[idx]).dot(np.abs(self.model.model.coef_).T)
        if self.batch_size > len(unlabeled_entry_ids):
            return list(idx)
        ask_ids = np.argpartition(evidence, -self.batch_size)[-self.batch_size:]
        return [unlabeled_entry_ids[idx[x]] for x in ask_ids]


class MinEvidenceSampling(QueryStrategy):
    def __init__(self, dataset, model, k=10, **kwargs):
        super(MinEvidenceSampling, self).__init__(dataset, **kwargs)
        self.model = model
        self.k = k
        self.model.train(self.dataset)

    def make_query(self):
        dataset = self.dataset
        self.model.train(dataset)
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        dvalue = self.model.predict_proba(X_pool)
        scores = np.min(dvalue, axis=1)
        if self.k:
            idx = scores.argsort()[-self.k:]
            evidence = np.abs(X_pool[idx]).dot(np.abs(self.model.model.coef_).T)
            ask_id = np.argmin(evidence)
            return unlabeled_entry_ids[idx[ask_id]]
        else:
            evidence = np.abs(X_pool).dot(np.abs(self.model.model.coef_).T)
            ask_id = np.argmin(evidence)
            return unlabeled_entry_ids[ask_id]


class ConflictingEvidenceSampling(QueryStrategy):
    def __init__(self, dataset, model, **kwargs):
        super(ConflictingEvidenceSampling, self).__init__(dataset, **kwargs)
        self.model = model
        self.model.train(self.dataset)

    def make_query(self):
        dataset = self.dataset
        self.model.train(dataset)
        unlabeled_entry_ids, x = dataset.get_unlabeled_entries()
        w = self.model.model.coef_
        b = self.model.model.intercept_
        evidence = np.abs(x).dot(np.abs(w).T) + np.abs(b) - np.abs(x.dot(w.T) + b)
        ask_id = np.argmax(evidence)

        return unlabeled_entry_ids[ask_id]


class RealLossSampling(QueryStrategy):
    def __init__(self, dataset, model, real_y, **kwargs):
        super(RealLossSampling, self).__init__(dataset, **kwargs)
        self.model = model
        self.y = real_y
        self.model.train(self.dataset)

    def make_query(self):
        dataset = self.dataset
        self.model.train(dataset)
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        dvalue = self.model.predict_proba(X_pool)
        loss = [1-dvalue[i, self.y[unlabeled_entry_ids[i]]] for i in range(len(dvalue))]

        ask_id = np.argmax(loss)

        return unlabeled_entry_ids[ask_id]


class SecondPassUncertaintySampling(QueryStrategy):
    def __init__(self, dataset, bp, model_name, name, **kwargs):
        super(SecondPassUncertaintySampling, self).__init__(dataset, **kwargs)
        with open(os.path.join(CODE_DIR, 'vectors', 'bow', name, model_name+'_9_{}.pkl'.format(str(bp).rjust(4, '0'))), 'rb') as f:
            self.best_score, self.model = pickle.load(f)
        self.ids, data = dataset.get_unlabeled_entries()
        self.unc = np.max(self.model.predict_proba(data), axis=1)
        self.d = {k: v for k,v in zip(self.ids, self.unc)}
        self.order = sorted(self.d.keys(), key=lambda x: self.d[x])

    def make_query(self):
        return self.order.pop(0)


class ChangeLabelUncertaintySampling(QueryStrategy):
    def __init__(self, dataset, bp, model_name, y_train, name, **kwargs):
        super(ChangeLabelUncertaintySampling, self).__init__(dataset, **kwargs)
        with open(os.path.join(CODE_DIR, 'vectors', 'bow', name, model_name+'_9_{}.pkl'.format(str(bp).rjust(4, '0'))), 'rb') as f:
            self.best_score, self.model = pickle.load(f)
        self.ids, data = dataset.get_unlabeled_entries()
        y = y_train[self.ids]
        self.ideal_proportion = 0.5
        self.positive = (sum(y_train)-sum(y))
        self.core_size = 10
        unc = np.max(self.model.predict_proba(data), axis=1)
        self.d = {k: v for k,v in zip(self.ids, unc)}

        i_0 = [i for i in self.ids if y_train[i] == 0]
        i_1 = [i for i in self.ids if y_train[i]]
        self.order_0 = sorted(i_0, key=lambda x: self.d[x])
        self.order_1 = sorted(i_1, key=lambda x: self.d[x])

    def make_query(self):
        to_pop = 1 if self.positive/self.core_size <= self.ideal_proportion else 0
        pop = [self.order_0, self.order_1]
        self.core_size += 1
        if len(pop[to_pop]) == 0:
            to_pop = (to_pop+1) % 2
        if len(pop[to_pop]) == 0:
            raise IndexError
        self.positive += to_pop
        return pop[to_pop].pop(0)

class Parser:
    all_datasets = ['cornell-sent-polarity', 'cornell-sent-subjectivity', 'ag_news2', 'ag_news3', 'dbpedia3', 'dbpedia8',
                    'ag_news_big2', 'ag_news_big3', 'dbpedia_big3', 'dbpedia_big8', ]
    encoder_name_to_class = {'fasttext': FastTextEncoder, 'glove': GloveEncoder, 'word2vec': Word2VecEncoder,
                             'bow': BoWExtractor, 'cbow': CBOWEncoder, 'transformer': TransformerEncoder}
    strategy_name_to_class = {'random': RandomSampling, 'unc': UncertaintySampling, 'secondpass': SecondPassUncertaintySampling,
                              'evidence100': MaxEvidenceSampling, 'evidence250': partial(MaxEvidenceSampling,k=250), 'evidence500': partial(MaxEvidenceSampling,k=500),
                              'loss': RealLossSampling, 'minEv': partial(MinEvidenceSampling, k=0), 'confEv': ConflictingEvidenceSampling}

    all_models = ['Regression', 'RegressionStable', 'svmLinear', 'svmHinge']
    all_regs = ['', '0,1', '0,01', '10', '50', '100', 'L1']
    all_models = [a+b for a,b in product(all_models, all_regs)]
    all_models.remove('svmHingeL1')
    all_models = all_models + ['RF']
    all_strategies = list(strategy_name_to_class.keys())
    all_encoders = list(encoder_name_to_class.keys())

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description='Hi')
        parser.add_argument('--name', type=str, required=False, default=Parser.all_datasets[:6],
                            help='dataset name. if not provided all datasets are used', nargs='+')
        parser.add_argument('--encoder', type=str, required=False, choices=Parser.all_encoders, default=Parser.all_encoders,
                            help='enc name. if not provided all enc are used', nargs='+')
        parser.add_argument('--strategy', type=str, required=False, choices=Parser.all_strategies, default='unc',
                            help='strategy name. if not provided all enc are used', nargs='+')
        parser.add_argument('--model', type=str, required=False, choices=Parser.all_models, default=Parser.all_models, nargs='+')
        parser.add_argument('--regs', type=str, required=False, choices=[str(x) for x in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]], default=[], nargs='+')
        parser.add_argument('--batch', type=float, required=False, default=0.5, help='batch size in % or samples')
        parser.add_argument('--loss', required=False, default=False, action='store_true')
        parser.add_argument('--downsample', required=False, default=-1)
        parser.add_argument('--round', type=int, required=False, default=0, help='start round')
        parser.add_argument('--vectors', required=False, default=False, action='store_true')
        args = parser.parse_args()
        names = args.name
        encoders = args.encoder
        strategy = args.strategy
        models = args.model
        if type(names) != list:
            names = [names]
        if type(encoders) != list:
            encoders = [encoders]
        if type(strategy) != list:
            strategy = [strategy]
        if type(models) != list:
            models = [models]
        if len(models) == 1 and len(args.regs):
            models = [models[0] + str(round(1/float(x), 2)).replace('.', ',') for x in args.regs]
        return names, encoders, strategy, args.round, models, float(args.downsample), args.vectors, args.batch


def model_factory(model_name):
    kwargs = {'max_iter': 20000, 'name': model_name}
    if model_name.endswith('L1'):
        kwargs['penalty'] = 'l1'
        kwargs['random_state'] = 1
    else:
        if not model_name[-1].isdigit():
            kwargs['C'] = 1.0
        else:
            first_digit = min([i for i in range(len(model_name)) if model_name[i].isdigit()])
            kwargs['C'] = float(model_name[first_digit:].replace(',', '.'))

    if model_name.startswith('Regression'):
        if not model_name.endswith('L1'):
            kwargs['solver'] = 'lbfgs'
        if model_name.startswith('RegressionStable'):
            kwargs['C'] *= 1000
            return LogisticRegressionStable(**kwargs)
        return LogisticRegression(**kwargs)

    elif model_name.startswith('svm'):
        kwargs['dual'] = False
        kwargs['random_state'] = 1
        if model_name.startswith('svmHinge'):
            kwargs['loss'] = 'hinge'
            kwargs['dual'] = True
        return LinearSVM(**kwargs)
    elif model_name.startswith('RF'):
        return SklearnProbaAdapter(RandomForestClassifier())


def get_loss(d, mn):
    if mn.startswith('svmLinear'):
        l = np.sum(np.max(1-d, 0) ** 2)
    elif mn.startswith('svmHinge'):
        l = np.sum(np.max(1-d, 0))
    else:
        l = np.log(1 + np.exp(-d)).sum()
    return l


def get_decision(ds, current_model):
    x_train, y_train = ds.get_labeled_entries()
    y_train = np.array(y_train)
    d = current_model.model.decision_function(x_train)
    return d * (2 * y_train - 1), d

