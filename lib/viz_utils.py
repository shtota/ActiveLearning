from feature_extraction import BoWExtractor, Word2VecEncoder
from data_loader import load_train_test

ALL_DATASETS = ['cornell-sent-polarity', 'cornell-sent-subjectivity', 'ag_news2', 'ag_news3', 'dbpedia3', 'dbpedia8']
DATASETS = ['cornell-sent-polarity', 'ag_news3', 'dbpedia3']
#BIG_DATASETS = ['ag_news2_big', 'ag_news3_big', 'dbpedia3_big', 'dbpedia8_big']
HUE_ORDER = ['0,1', '1', '10', 'L1(C=1)']


def get_baseline(ds):
    #if ds in classification_dataset_names:
        #return metadata[ds][0]
    encoder = BoWExtractor()
    _, X_train, X_test, y_train, y_test = load_train_test(ds, encoder, -1)
    prop = sum(y_test)/len(y_test)
    return max(prop, 1-prop)


def get_labels(ds):
    encoder = BoWExtractor()
    _, X_train, X_test, y_train, y_test = load_train_test(ds, encoder, -1)
    return y_train


def reg_map(x):
    if x.endswith('L1'):
        return 'L1(C=1)'
    if not x[-1].isdigit():
        c = 1
    else:
        first_digit = min([i for i in range(len(x)) if x[i].isdigit()])
        c = float(x[first_digit:].replace(',', '.'))
        c = 1/c
    if x.startswith('RegressionStable'):
        c = c/1000
    if int(c) == c:
        return str(int(c))
    return str(round(c,5)).replace('.',',')


def model_type(x):
    if x.startswith('Regression'):
        if x.startswith('RegressionStable'):
            return 'RegressionStable'
        return 'Regression'
    if x.startswith('svmLinear'):
        return 'svmSquared'
    return 'svm'