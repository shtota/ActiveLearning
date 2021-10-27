import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import time
from config import CODE_DIR
import pickle
from data_loader import load_train_test
from libact.base.dataset import Dataset
from utils import Parser, model_factory, get_decision, get_loss
from sklearn.metrics import f1_score, roc_auc_score
from math import ceil
START_SIZE = 10
RESTARTS = 10


class ActiveLearner:
    def __init__(self, X_train, y_train, X_test, y_test, name: str, encoder_name: str,
                 record_vectors: bool, batch_size=0.5):
        self.X_train = X_train
        self.y_train = y_train
        self.train_ds = None
        self.pool_ds = None
        self.test_ds = Dataset(X_test, y_test, 'test')
        self.dataset_name = name
        self.encoder_name = encoder_name
        self.record_vectors = record_vectors
        self.batch_size = batch_size

        self.model = None

    def initiate_core_set(self, seed):
        idx = self._create_round_labels(seed)
        round_labels = np.array([None] * len(self.y_train))
        round_labels[idx] = self.y_train[idx]
        self.train_ds = Dataset(self.X_train, round_labels, 'train')

        missing_idx = [i for i,x in enumerate(round_labels) if x is None]
        round_labels = np.array([None] * len(self.y_train))
        round_labels[missing_idx] = self.y_train[missing_idx]
        self.pool_ds = Dataset(self.X_train, round_labels, 'pool')

    def run(self, strategy_name, model_name, start_round=0):
        for i in range(start_round, RESTARTS):
            self.run_one_round(strategy_name, model_name, i)

    def _create_round_labels(self, round_):
        random = np.random.RandomState(round_)
        idx = random.choice(len(self.y_train), START_SIZE, False)
        while len(set(self.y_train[idx])) == 1:
            idx = random.choice(len(self.y_train), START_SIZE, False)
        return idx

    def log_model_metrics(self, step_results):
        step_results['regularization'] = (np.sum(self.model.model.coef_**2) + self.model.model.intercept_**2)[0]/2
        for ds in [self.test_ds, self.train_ds, self.pool_ds]:
            X, y = ds.get_labeled_entries()
            y = np.array(y, dtype=bool)
            if len(y):
                decision, predictions = get_decision(ds, self.model)
                step_results['{}_loss'.format(ds.name)] = get_loss(decision, self.model.name)
                step_results['{}_accuracy'.format(ds.name)] = sum([x > 0 for x in decision]) / len(decision)
                step_results['{}_f1'.format(ds.name)] = f1_score(y, predictions >= 0)
                step_results['{}_ROC'.format(ds.name)] = roc_auc_score(y, predictions)
    
                if ds == self.train_ds:
                    step_results['n_support_vecs'] = sum(decision <= 1)

                # Class confidence
                step_results['{}_margin_1'.format(ds.name)] = predictions[y].mean()
                step_results['{}_margin_0'.format(ds.name)] = -predictions[~y].mean()

    def log_batch_metrics(self, ask_ids, step_results):
        X = self.X_train[ask_ids]
        y = self.y_train[ask_ids]
        probabilities = self.model.predict_proba(X)
        step_results['queried_certainty'] = np.max(probabilities, axis=1).mean()
        loss = 0
        for i, label in enumerate(y):
            loss += 1 - probabilities[i, label]
        step_results['queried_loss'] = loss/len(y)
        step_results['ask_ids'] = ';'.join([str(x) for x in ask_ids])

        # TODO: stopping criteria

    def run_one_round(self, strategy_name, model_name, round_):
        self.initiate_core_set(round_)
        self.model = model_factory(model_name)
        self.model.train(self.train_ds)
        strategy = Parser.strategy_name_to_class[strategy_name](self.train_ds, model=self.model, real_y=self.y_train,
                                                                train_on_query=False, batch_size=self.batch_size,
                                                                random_state=round_)

        round_results = []
        batches = ceil(sum(self.pool_ds.get_labeled_mask())/self.batch_size)
        description = '{} {} {} {} {}'.format(round_, strategy_name, self.dataset_name, self.encoder_name, model_name)

        for batch_no in tqdm(range(batches), description, batches):
            step_results = {'core_size': START_SIZE + self.batch_size*batch_no}
            self.log_model_metrics(step_results)

            ask_ids = strategy.make_query() # TODO: support for train=False, support for batches
            self.log_batch_metrics(ask_ids, step_results) # Batch accuracy, batch uncertainty, batch loss,

            for ask_id in ask_ids:
                self.train_ds.update(ask_id, self.y_train[ask_id])
                self.pool_ds.update(ask_id, None)
            self.model.train(self.train_ds)

            round_results.append(step_results)

        step_results = dict(round_results[-1]) # to avoid NaNs in batch metric after full training I copy them from the last iteration
        step_results['core_size'] = self.X_train.shape[0]
        self.log_model_metrics(step_results)
        round_results.append(step_results)

        results = pd.DataFrame(data=round_results)
        results.to_csv(os.path.join(CODE_DIR, 'results', self.encoder_name, self.dataset_name,
                                    strategy_name + '_' + str(round_) + '_' + model_name + '.csv'), index=False)

    def dump_vector(self, model_name, round_, core_size, test_score, current_model):
        os.makedirs(os.path.join(CODE_DIR, 'vectors', self.encoder_name, self.dataset_name), exist_ok=True)
        with open(os.path.join(CODE_DIR, 'vectors', self.encoder_name, self.dataset_name,
                               model_name + '_' + str(round_) + '_' + str(core_size).rjust(4, '0') + '.pkl'),
                  'wb') as f:
            pickle.dump((test_score, current_model), f)


def main():
    names, encoders, strategies, start_round, models, downsample, vecs, batch_size = Parser.parse_args()
    a = time.time()
    for encoder_name in encoders:
        for dataset_name in names:
            encoder = Parser.encoder_name_to_class[encoder_name]()
            dataset_name, X_train, X_test, y_train, y_test = load_train_test(dataset_name, encoder, downsample)
            encoder.clear()
            if batch_size < 1:
                batch_size = ceil(X_train.shape[0]*0.01*batch_size)
            learner = ActiveLearner(X_train, y_train, X_test, y_test, dataset_name, encoder_name, vecs, batch_size)
            for model_name in models:
                for strategy_name in strategies:
                    os.makedirs(os.path.join(CODE_DIR, 'results', encoder_name, dataset_name), exist_ok=True)
                    learner.run(strategy_name, model_name, start_round=start_round)
                    pass

    print('done in ', time.time()-a)


if __name__ == '__main__':
    #runtime_check()
    main()

