import numpy as np
import math
from libact.base.interfaces import ProbabilisticModel
from libact.models import *
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR


class LogisticRegressionStable(ProbabilisticModel):

    def __init__(self, C, *args, **kwargs):
        self.C = C
        self.name = kwargs.pop('name', '')
        self.model = LR(*args, **kwargs)
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


def model_factory(model_name):
    kwargs = {'max_iter': 200000, 'name': model_name}
    if model_name.endswith('L1'):
        kwargs['penalty'] = 'l1'
        kwargs['random_state'] = 1
        kwargs['C'] = 1
    else:
        if not model_name[-1].isdigit():
            kwargs['C'] = 1.0
        else:
            first_digit = min([i for i in range(len(model_name)) if model_name[i].isdigit()])
            kwargs['C'] = float(model_name[first_digit:].replace(',', '.'))

    if model_name.startswith('Regression'):
        if model_name.endswith('L1'):
            kwargs['solver'] = 'liblinear'
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


def get_all_models():
    all_models = ['Regression', 'RegressionStable', 'svmLinear', 'svmHinge']
    all_regs = ['', '0,1', '10', 'L1']
    all_models = [a + b for a, b in product(all_models, all_regs)]
    all_models.remove('svmHingeL1')
    all_models.remove('RegressionStableL1')
    #all_models = all_models + ['RF']
    return all_models


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


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


sigmoid = np.vectorize(_sigmoid)

