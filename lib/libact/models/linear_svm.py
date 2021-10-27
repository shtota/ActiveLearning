"""SVM

An interface for scikit-learn's C-Support Vector Classifier model.
"""
import logging
LOGGER = logging.getLogger(__name__)

import numpy as np
from sklearn.svm import LinearSVC

from libact.base.interfaces import ProbabilisticModel


class LinearSVM(ProbabilisticModel):

    """C-Support Vector Machine Classifier

    When decision_function_shape == 'ovr', we use OneVsRestClassifier(SVC) from
    sklearn.multiclass instead of the output from SVC directory since it is not
    exactly the implementation of One Vs Rest.

    References
    ----------
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """

    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop('name', '')
        self.model = LinearSVC(*args, **kwargs)

    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)

    def predict_proba(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        proba = np.exp(dvalue)
        proba = proba/(1+proba)
        return np.vstack((proba, 1-proba)).T