from libact.base.interfaces import QueryStrategy
import numpy as np
from libact.query_strategies import UncertaintySampling, RandomSampling
from functools import partial


class MaxEvidenceSampling(QueryStrategy):
    def __init__(self, dataset, model, k=2, **kwargs):
        super(MaxEvidenceSampling, self).__init__(dataset, **kwargs)
        self.model = model
        self.k = k
        self.model.train(self.dataset)

    def make_query(self):
        dataset = self.dataset
        if self.train_on_query:
            self.model.train(dataset)
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        if self.batch_size > len(unlabeled_entry_ids):
            return unlabeled_entry_ids
        uncertainty = np.min(self.model.predict_proba(X_pool), axis=1)
        idx = uncertainty.argsort()[-self.k*self.batch_size:]
        evidence = np.abs(X_pool[idx]).dot(np.abs(self.model.model.coef_).T)
        ask_ids = np.argpartition(evidence, -self.batch_size)[-self.batch_size:]
        return [unlabeled_entry_ids[idx[x]] for x in ask_ids]


class MinEvidenceSampling(QueryStrategy):
    def __init__(self, dataset, model, **kwargs):
        super(MinEvidenceSampling, self).__init__(dataset, **kwargs)
        self.model = model
        self.model.train(self.dataset)

    def make_query(self):
        dataset = self.dataset
        if self.train_on_query:
            self.model.train(dataset)
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        if self.batch_size > len(unlabeled_entry_ids):
            return unlabeled_entry_ids
        evidence = np.abs(X_pool).dot(np.abs(self.model.model.coef_).T)
        ask_ids = np.argpartition(evidence, self.batch_size)[:self.batch_size]
        return [unlabeled_entry_ids[x] for x in ask_ids]


class ConflictingEvidenceSampling(QueryStrategy):
    def __init__(self, dataset, model, **kwargs):
        super(ConflictingEvidenceSampling, self).__init__(dataset, **kwargs)
        self.model = model
        self.model.train(self.dataset)

    def make_query(self):
        dataset = self.dataset
        if self.train_on_query:
            self.model.train(dataset)
        unlabeled_entry_ids, x = dataset.get_unlabeled_entries()
        w = self.model.model.coef_
        b = self.model.model.intercept_
        evidence = np.abs(x).dot(np.abs(w).T) + np.abs(b) - np.abs(x.dot(w.T) + b)
        ask_ids = np.argpartition(evidence, -self.batch_size)[-self.batch_size:]
        return [unlabeled_entry_ids[x] for x in ask_ids]


class RealLossSampling(QueryStrategy):
    def __init__(self, dataset, model, real_y, **kwargs):
        super(RealLossSampling, self).__init__(dataset, **kwargs)
        self.model = model
        self.y = real_y
        self.model.train(self.dataset)

    def make_query(self):
        dataset = self.dataset

        if self.train_on_query:
            self.model.train(dataset)
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        if self.batch_size > len(unlabeled_entry_ids):
            return list(unlabeled_entry_ids)
        dvalue = self.model.predict_proba(X_pool)
        loss = [1-dvalue[i, self.y[unlabeled_entry_ids[i]]] for i in range(len(dvalue))]

        ask_ids = np.argpartition(loss, -self.batch_size)[-self.batch_size:]
        return [unlabeled_entry_ids[x] for x in ask_ids]


class SecondPassUncertaintySampling(QueryStrategy):
    def __init__(self, dataset, bp, model_name, name, **kwargs):
        raise NotImplementedError
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
        raise NotImplementedError
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


STRATEGY_NAME_TO_CLASS = {'random': RandomSampling,
                          'unc': UncertaintySampling,
                          'maxEv2': partial(MaxEvidenceSampling, k=2),
                          'maxEv5': partial(MaxEvidenceSampling, k=5),
                          'loss': RealLossSampling,
                          'minEv': partial(MinEvidenceSampling, k=0),
                          'confEv': ConflictingEvidenceSampling,
                          # 'secondpass': SecondPassUncertaintySampling
                          }

ALL_STRATEGIES = sorted(STRATEGY_NAME_TO_CLASS.keys())