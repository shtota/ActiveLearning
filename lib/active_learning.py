import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from config import CODE_DIR
import pickle
from libact.base.dataset import Dataset
from utils import get_decision, get_loss, sigmoid, model_factory
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score
from my_strategies import STRATEGY_NAME_TO_CLASS
from math import ceil
START_SIZE = 30
RESTARTS = 10


class ActiveLearner:
    def __init__(self, X_train, y_train, X_test, y_test, name: str, embedding_name: str, batch_size=0.5, skip_existing=False):
        self.X_train = X_train
        self.y_train = y_train
        self.core_ds = None
        self.pool_ds = None
        self.test_ds = Dataset(X_test, y_test, 'test')
        self.dataset_name = name
        self.embedding_name = embedding_name
        if batch_size < 1:
            batch_size = ceil(X_train.shape[0] * 0.01 * batch_size)
        self.batch_size = batch_size
        self.skip_existing = skip_existing

        self.model = None
        self.previous_predictions = {}

        self.strategy_name = ''
        self.model_name = ''
        self.round = 0

    def run(self, strategy_name, model_name, start_round=0):
        self.strategy_name = strategy_name
        self.model_name = model_name
        self.round = start_round
        for self.round in range(self.round, RESTARTS):
            if self.skip_existing and os.path.exists(self.csv_path()):
                print('Skipping', self.csv_path())
                continue
            try:
                self.initiate_core_set()
                self.run_one_round()
            except KeyboardInterrupt:
                print('Received interrupt')
                exit(0)
            except Exception as e:
                raise e
                with open('./errors.log', 'a') as f:
                   f.write('{} {} {} {} {} {}'.format(self.dataset_name, self.embedding_name,
                                                      self.strategy_name, self.model_name, self.round, e))
                continue

    def run_one_round(self):
        self.model = model_factory(self.model_name, self.embedding_name)
        self.model.train(self.core_ds)
        strategy_class = STRATEGY_NAME_TO_CLASS[self.strategy_name]
        strategy = strategy_class(self.core_ds, model=self.model, real_y=self.y_train,
                                  train_on_query=False, batch_size=self.batch_size, random_state=self.round)

        round_results = list()
        batches = ceil(sum(self.pool_ds.get_labeled_mask())/self.batch_size)
        description = '{} {} {} {} {}'.format(self.round, self.strategy_name, self.dataset_name, self.embedding_name, self.model_name)

        for batch_no in tqdm(range(batches+1), description, batches):
            if batch_no:
                step_results = dict(round_results[-1])
            else:
                step_results = {}
            step_results['core_size'] = self.core_ds.len_labeled()
            self.log_model_metrics(step_results)
            if step_results['core_size'] < self.X_train.shape[0]:
                ask_ids = strategy.make_query()
                self.log_batch_metrics(ask_ids, step_results) # Batch accuracy, batch uncertainty, batch loss,

                for ask_id in ask_ids:
                    self.core_ds.update(ask_id, self.y_train[ask_id])
                    self.pool_ds.update(ask_id, None)
                self.model.train(self.core_ds)

            round_results.append(step_results)

        results = pd.DataFrame(data=round_results)
        results.to_csv(self.csv_path(), index=False)

    def csv_path(self):
        return os.path.join(CODE_DIR, 'results', self.embedding_name, self.dataset_name,
                            self.strategy_name + '_' + str(self.round) + '_' + self.model_name + '.csv')

    def initiate_core_set(self):
        idx = self._create_round_labels()
        round_labels = np.array([None] * len(self.y_train))
        round_labels[idx] = self.y_train[idx]
        self.core_ds = Dataset(self.X_train, round_labels, 'train')

        missing_idx = [i for i,x in enumerate(round_labels) if x is None]
        round_labels = np.array([None] * len(self.y_train))
        round_labels[missing_idx] = self.y_train[missing_idx]
        self.pool_ds = Dataset(self.X_train, round_labels, 'pool')

    def _create_round_labels(self):
        random = np.random.RandomState(self.round)
        idx = random.choice(len(self.y_train), START_SIZE, False)
        while len(set(self.y_train[idx])) == 1:
            idx = random.choice(len(self.y_train), START_SIZE, False)
        return idx

    def log_model_metrics(self, step_results):
        step_results['regularization'] = (np.sum(self.model.model.coef_ ** 2) + self.model.model.intercept_ ** 2)[0] / 2
        for ds in [self.test_ds, self.core_ds, self.pool_ds]:
            X, y = ds.get_labeled_entries()
            y = np.array(y, dtype=bool)
            if len(y) == 0:
                continue
            decision, distances = get_decision(ds, self.model)  # w*x*y and w*x accordingly. y in {-1, 1}
            y_pred = distances >= 0
            step_results['{}_majority'.format(ds.name)] = max(sum(y)/len(y), 1 - sum(y)/len(y))
            step_results['{}_loss'.format(ds.name)] = get_loss(decision, self.model.name)
            step_results['{}_accuracy'.format(ds.name)] = sum([x > 0 for x in decision]) / len(decision)
            step_results['{}_certainty'.format(ds.name)] = sigmoid(np.abs(decision)).mean()
            self.get_kappa(ds, step_results, y_pred)
            self.previous_predictions[ds.name] = [(original_index, y_pred[i])
                                                  for i, original_index
                                                  in enumerate(np.where(ds.get_labeled_mask())[0])]
            if ds == self.core_ds:
                step_results['n_support_vecs'] = sum(decision <= 1)
                step_results['farthest_core'] = np.max(decision)

            if ds == self.pool_ds:
                step_results['closest_pool'] = np.min(np.abs(decision))

            if len(np.unique(y)) == 1:
                continue

            # Class confidence
            step_results['{}_margin_1'.format(ds.name)] = distances[y].mean()
            step_results['{}_margin_0'.format(ds.name)] = -distances[~y].mean()
            step_results['{}_f1'.format(ds.name)] = f1_score(y, y_pred)
            step_results['{}_ROC'.format(ds.name)] = roc_auc_score(y, distances)

    def log_batch_metrics(self, ask_ids, step_results):
        X = self.X_train[ask_ids]
        y = self.y_train[ask_ids]
        probabilities = self.model.predict_proba(X)
        step_results['mean_queried_certainty'] = np.max(probabilities, axis=1).mean()
        step_results['max_queried_certainty'] = np.max(probabilities)
        variance = (np.max(probabilities, axis=1) - step_results['mean_queried_certainty'])
        variance = variance.dot(variance.T) / len(variance)
        step_results['variance_queried_certainty'] = variance

        loss = 0
        for i, label in enumerate(y):
            loss += 1 - probabilities[i, label]
        step_results['queried_loss'] = loss / len(y)
        step_results['ask_ids'] = ';'.join([str(x) for x in ask_ids])

    def get_kappa(self, ds, step_results, y_pred):
        step_results['{}_kappa_agreement'.format(ds.name)] = 1
        if ds.name not in self.previous_predictions.keys():
            return

        previous_y = np.array([x[1] for x in self.previous_predictions[ds.name]])

        if ds == self.test_ds:
            if all(y_pred == previous_y):
                return
            step_results['kappa_agreement_{}'.format(ds.name)] = cohen_kappa_score(y_pred, previous_y, labels=[0, 1])
            return

        relevant_idx = {x[0] for x in self.previous_predictions[ds.name]}.intersection(set(np.where(ds.get_labeled_mask())[0]))
        relevant_old_y = np.array([y for index, y in self.previous_predictions[ds.name] if index in relevant_idx])
        relevant_y = np.array([y for index, y in zip(np.where(ds.get_labeled_mask())[0], y_pred) if index in relevant_idx])
        if all(relevant_y == relevant_old_y):
            return
        step_results['kappa_agreement_{}'.format(ds.name)] = cohen_kappa_score(relevant_old_y, relevant_y, labels=[0, 1])

    def dump_vector(self, model_name, round_, core_size, test_score, current_model):
        # TODO: consider dumping vectors at each step
        os.makedirs(os.path.join(CODE_DIR, 'vectors', self.embedding_name, self.dataset_name), exist_ok=True)
        with open(os.path.join(CODE_DIR, 'vectors', self.embedding_name, self.dataset_name,
                               model_name + '_' + str(round_) + '_' + str(core_size).rjust(4, '0') + '.pkl'),
                  'wb') as f:
            pickle.dump((test_score, current_model), f)

    def _test(self):
        if self.embedding_name == 'bow':
            from scipy.sparse.linalg import norm
        else:
            from numpy.linalg import norm
        from sklearn.preprocessing import normalize
        scores = []
        for i in ['0,01', '0,1', '1', '10', '100']:
            self.model = model_factory('Regression{}'.format(i))
            self.model.model.fit(self.X_train, self.y_train)
            scores.append(round(self.model.score(self.test_ds), 2))
        print(self.dataset_name, scores)
        scores = []

        self.X_train = normalize(self.X_train, norm='l1', axis=1)
        for i in ['0,01', '0,1', '1', '10', '100']:
            self.model = model_factory('Regression{}'.format(i))
            self.model.model.fit(self.X_train, self.y_train)
            scores.append(round(self.model.score(self.test_ds), 2))
        print(self.dataset_name, scores)

        return
