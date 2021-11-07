import pandas as pd
import os
import numpy as np
from config import CODE_DIR
from sklearn.model_selection import train_test_split
import json
from scipy.sparse.linalg import norm

TARGET_COLUMN_NAME = 'target'
CORE_SET_SIZE = 10
FEATURES_COLUMN_NAME = 'features'
DATABASE_PARAMS = {
    'codementor_SA': {'header': None, 'delimiter': '\t'},
    'cornell-sent-polarity': {'header': None, 'delimiter': '\t', 'encoding': 'latin-1'},
    'cornell-sent-subjectivity': {'header': None, 'delimiter': '\t', 'encoding': 'latin-1'}
}
WEIRD_DATASETS = {'pang04', 'pang05', 'pang04_biased', 'pang05_biased', 'mcauley15', 'mcauley15_biased', 'mcauley15_balanced','mcauley15_balanced_biased'}

data_folder_abspath = os.path.abspath(os.path.join(CODE_DIR, 'experiment_data'))


def encoded_filename(name):
    return os.path.join(data_folder_abspath, name, 'encoded_data.csv')


def data_filename(name, encoder):
    s = 'all'
    if encoder is not None:
        s += '_' + str(encoder)
        s += '.np'
    else:
        s += '_data.csv'

    return os.path.join(data_folder_abspath, name, s)


def _load_df(name):
    params = DATABASE_PARAMS.get(name, {'header': None})
    df = pd.read_csv(data_filename(name, None), **params)
    df.columns = ['target', 'text']
    return df


def load_features_labels(name, encoder):
    df = _load_df(name)
    labels = df['target'].values
    filename = data_filename(name, encoder)
    if os.path.exists(filename):
        features = np.loadtxt(filename)
        #print(np.mean(np.linalg.norm(features, axis=0)))
        return features, labels
    features = encoder.prepare_features(df['text'])
    #print(np.mean(norm(features, axis=0)))
    if type(features) == np.array or type(features) == np.ndarray:
        np.savetxt(filename, features)
    return features, labels


def clear_data(rows):
    labels = []
    texts = []
    for row in rows:
        if not row:
            continue
        label = int(row[-2])
        text = row[2:-5]
        labels.append(label)
        texts.append(text)
    return texts, labels

def load_texts(name):
    if name in WEIRD_DATASETS:
        res = []
        with open(os.path.join(data_folder_abspath, name, 'train_biased.txt'), 'r') as f:
            raw_data = f.read().split('\n')
            texts, labels = clear_data(raw_data)
        res += texts
        with open(os.path.join(data_folder_abspath, name, 'test.txt'), 'r') as f:
            raw_data = f.read().split('\n')
            texts, labels = clear_data(raw_data)
        res += texts
        return res
    else:
        df = _load_df(name)
        return list(df['text'])

def load_train_test(name, encoder, downsample, train_size=0.6):
    if name in WEIRD_DATASETS:
        if name.startswith('mcauley'):
            return load_mcauley(encoder, name.endswith('biased'), bool(name.count('balanced')))
        folder_name = name
        filename = 'train_biased.txt'
        if name.endswith('biased'):
            folder_name = name[:-len('_biased')]
        else:
            filename = 'train.txt'

        with open(os.path.join(data_folder_abspath, folder_name, filename), 'r') as f:
            raw_data = f.read().split('\n')
            texts, labels = clear_data(raw_data)

        X_train = encoder.prepare_features(texts)
        y_train = labels

        with open(os.path.join(data_folder_abspath, folder_name, 'test.txt'), 'r') as f:
            raw_data = f.read().split('\n')
            texts, labels = clear_data(raw_data)

        X_test = encoder.prepare_features(texts, fit=False)
        y_test = labels

    else:
        features, labels = load_features_labels(name, encoder)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=train_size, random_state=1)
        if downsample != -1 and name.startswith('dbpedia'):
            proportion = downsample
            random = np.random.RandomState(2)
            n_1 = sum(y_train)
            n_2 = int(n_1/(1-proportion)*proportion)
            n_drop = len(y_train) - n_1 - n_2
            candidates = [i for i, x in enumerate(y_train) if not x]
            to_drop = set(random.choice(candidates, n_drop, replace=False))
            to_stay = [i for i in range(len(y_train)) if i not in to_drop]

            X_train = X_train[to_stay]
            #X_train = vstack([X_train.getrow(i) for i in to_stay])
            y_train = np.array(y_train)[to_stay]
            name = name + '_downsampled' + str(proportion)

    #n = norm(X_train, axis=1)**2
    #mean_norm = np.mean(n)
    #print(name, mean_norm, 1/mean_norm)

    return name, X_train, X_test, y_train, y_test

def load_mcauley(encoder, biased=False, balanced=False):
    TEST_SIZE = 6500
    X_test = []
    y_test = []
    used_ind = []

    with open(os.path.join(data_folder_abspath, 'mcauley15', 'test.txt'), 'r') as f:
        raw_data = f.read().split('\n')
        texts, labels = clear_data(raw_data)
    if balanced:
        for label in [0, 1]:
            ind = [i for i, v in enumerate(labels) if v == label]
            ind = np.random.RandomState(1).choice(ind, int(TEST_SIZE / 2), replace=False)
            used_ind += list(ind)
    else:
        used_ind = np.random.RandomState(1).choice(range(len(labels)), TEST_SIZE, replace=False)

    for i in used_ind:
        X_test.append(texts[i])
        y_test.append(labels[i])
    used_ind = set(used_ind)
    remaining_indices = [i for i in range(len(labels)) if i not in used_ind]

    TRAIN_SIZE = 10000
    if biased:
        with open(os.path.join(data_folder_abspath, 'mcauley15', 'train_biased.txt'), 'r') as f:
            raw_data = f.read().split('\n')
            texts, labels = clear_data(raw_data)
            if balanced:
                X_train = []
                y_train = []
                for label in [0,1]:
                    ind = [i for i in range(len(labels)) if labels[i] == label]
                    ind = np.random.RandomState(1).choice(ind, int(TRAIN_SIZE/2), replace=False)
                    for i in ind:
                        X_train.append(texts[i])
                        y_train.append(labels[i])
                X_train = encoder.prepare_features(X_train)
            else:
                X_train = encoder.prepare_features(texts[:TRAIN_SIZE])
                y_train = labels[:TRAIN_SIZE]
    else:
        X_train = []
        y_train = []
        if balanced:
            for label in [0, 1]:
                ind = [i for i in remaining_indices if labels[i] == label]
                ind = np.random.RandomState(1).choice(ind, int(TRAIN_SIZE / 2), replace=False)
                for i in ind:
                    X_train.append(texts[i])
                    y_train.append(labels[i])
        else:
            ind = sorted(remaining_indices)
            ind = np.random.RandomState(1).choice(ind, TRAIN_SIZE, replace=False)
            for i in ind:
                X_train.append(texts[i])
                y_train.append(labels[i])
        X_train = encoder.prepare_features(X_train)

    X_test = encoder.prepare_features(X_test, fit=False)
    return X_train, X_test, y_train, y_test