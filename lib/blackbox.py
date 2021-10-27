import pandas as pd

import seaborn as sns
import numpy as np

from data_loader import load_train_test
from feature_extraction import FastTextEncoder, GloveEncoder, BoWExtractor, CBOWEncoder, Word2VecEncoder
from sklearn import linear_model
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
import argparse
import time
from scipy import sparse
import matplotlib.pyplot as plt
import os

all_datasets = ['stanford_SA', 'cornell-sent-polarity', 'cornell-sent-subjectivity', 'ag_news2', 'ag_news3',
                'dbpedia3', 'dbpedia8', 'pang04', 'pang05', 'pang04_biased', 'pang05_biased', 'mcauley15', 'mcauley15_biased',
                'mcauley15_balanced','mcauley15_balanced_biased', ]
encoder_name_to_class = {'fasttext': FastTextEncoder, 'glove': GloveEncoder, 'word2vec': Word2VecEncoder,
                         'bow': BoWExtractor, 'cbow': CBOWEncoder}
all_encoders = list(encoder_name_to_class.keys())


def parse_args():
    parser = argparse.ArgumentParser(description='Hi')
    parser.add_argument('--name', type=str, required=False, choices=all_datasets, default=all_datasets,
                        help='dataset name. if not provided all datasets are used', nargs='+')
    parser.add_argument('--encoder', type=str, required=False, choices=all_encoders, default=all_encoders,
                        help='enc name. if not provided all enc are used', nargs='+')
    args = parser.parse_args()
    names = args.name
    encoders = args.encoder
    if type(names) != list:
        names = [names]
    if type(encoders) != list:
        encoders = [encoders]
    return names, encoders


def neighbor_consistency_score(cluster, y, predicted, center):
    score = 0
    for j, r in enumerate(cluster):
        distance = sparse.linalg.norm(r - center)
        weight = np.exp(-distance ** 3 * 100)
        weight = weight * ((y[j] == predicted[j]) * 2 - 1)
        score += weight
    return score


def relative_score(cluster, y, predicted, center):
    score = 0
    for j, r in enumerate(cluster):
        distance = sparse.linalg.norm(r - center)
        weight = np.exp(-distance ** 3 * 100)
        weight = weight * ((y[0] == predicted[j]) * 2 - 1)
        score += weight
    return score


def relative_filtered_score(cluster, y, predicted, center):
    score = 0
    for j, r in enumerate(cluster):
        distance = sparse.linalg.norm(r - center)
        weight = np.exp(-distance ** 3 * 100)
        weight = (y[j] == predicted[j])*weight*((y[0] == predicted[j]) * 2 - 1)
        score += weight
    return score


def score_clusters(X, y, predicted, all_distances, clustering_method, scoring_functions):
    K = 20
    length = X.shape[0]
    clusters = []
    scores = [[] for x in scoring_functions]
    for i in range(length):
        indices = sorted(range(length), key=lambda x: all_distances[i, x])[:K]
        rows = [X.getrow(j) for j in indices]
        if clustering_method == 'mean':
            point = sum(rows)/len(indices)
        else:
            point = rows[0]
        clusters.append(point)
        for j, f in enumerate(scoring_functions):
            inputs = (rows, y[indices], predicted[indices], point)
            scores[j].append(f(*inputs))

        if i%1000 == 99:
            print(i+1)
    return sparse.vstack(clusters), scores


def visualize(df, name, precision=1):
    count_to_size = {50: 50, 100: 150, 500: 300, 1000: 600, 2000: 1000, 2001: 1400}

    def size_grouping(x):
        for s in [50, 100, 500, 1000, 2000]:
            if x < s:
                return '< ' + str(s)
        return '>= 2000'

    string_to_size = {size_grouping(x - 1): count_to_size[x] for x in [50, 100, 500, 1000, 2000, 2001]}

    df['confidence_bin'] = (df.confidence*precision).round(0) / precision
    data = df.groupby(['confidence_bin', 'cluster_scoring', 'confidence_model']).mean()['accuracy'].reset_index()
    data['size'] = df.groupby(['confidence_bin', 'cluster_scoring', 'confidence_model']).count()['accuracy'].values
    data['size'] = data['size'].map(lambda x: size_grouping(x))

    sns.set()
    plt.figure(figsize=(24, 12))
    plt.ylim(-0.05, 1.05)
    sns.set()
    sns.scatterplot(x='confidence_bin', y='accuracy', data=data, size='size', sizes=string_to_size,
                    hue='cluster_scoring', size_order=string_to_size.keys(), style='confidence_model', style_order=['linear', 'closest_clusters'],
                    alpha=0.8)
    plt.title(name + '. Confidence approximation with model')
    baseline = df.accuracy.mean()
    naive_baseline = max(df.labels.mean(), 1-df.labels.mean())
    plt.plot([df.confidence_bin.min() - 1, df.confidence_bin.max() + 1], [baseline, baseline], linewidth=1,
             color='green')
    plt.plot([df.confidence_bin.min() - 1, df.confidence_bin.max() + 1], [naive_baseline, naive_baseline], linewidth=1,
             color='black')
    plt.legend(prop={'size': 20}, fancybox=True, framealpha=0.3)

    if os.path.exists('confidence/'+name + '_confidence.png'):
        os.remove('confidence/'+name + '_confidence.png')
    plt.savefig('confidence/'+name + '_confidence.png')

    #plt.show()
    df = df[df.cluster_scoring == df.cluster_scoring.unique()[0]]
    uncertaincy = np.max(np.array([df['predicted'], 1 - df.predicted]), axis=0)
    df['uncertaincy_bin'] = (uncertaincy * 20).round(0) / 20
    data = df.groupby(['uncertaincy_bin']).mean()['accuracy'].reset_index()
    data['size'] = df.groupby(['uncertaincy_bin']).count()['accuracy'].values
    data['size'] = data['size'].map(lambda x: size_grouping(x))

    plt.figure(figsize=(24, 12))
    plt.ylim(-0.05, 1.05)
    sns.set()
    sns.scatterplot(x='uncertaincy_bin', y='accuracy', data=data, size='size', sizes=string_to_size,
                    size_order=string_to_size.keys())
    plt.title(name + '. Confidence approximation with model\'s uncertaincy')
    plt.legend(prop={'size': 20},fancybox=True, framealpha=0.5)
    plt.plot([0.45, 1.05], [baseline, baseline], linewidth=1, color='green')
    plt.plot([0.45, 1.05], [naive_baseline, naive_baseline], linewidth=1, color='black')
    if os.path.exists('confidence/'+name + '_uncertaincy.png'):
        os.remove('confidence/'+name + '_uncertaincy.png')
    plt.savefig('confidence/'+name + '_uncertaincy.png')
    #plt.show()

def visualize_weights(prediction_model, models, name):
    plt.figure(figsize=(10,10))
    X = []
    y = []
    names = []
    for model_name, model in models.items():
        X += list(abs(prediction_model.coef_[0]))
        y += list(model.coef_)
        names += [model_name]*len(model.coef_)
    sns.set()
    df = pd.DataFrame(data=zip(X,y, names), columns=['abs weight in classifier', 'weight in scoring model', 'scoring model'])
    sns.scatterplot(x='abs weight in classifier', y='weight in scoring model', data=df,
                    style='scoring model', hue='scoring model', markers=True, size='scoring model', sizes=[10,10,10], alpha=0.5)
    if os.path.exists('confidence/'+name + '_weights.png'):
        os.remove('confidence/'+name + '_weights.png')
    plt.title(name)
    plt.savefig('confidence/'+name + '_weights.png')
    #plt.show()
TEST = False
TEST_SIZE = 1500
GMM = True

def experiment(name):
    methods = {'point': [neighbor_consistency_score, relative_score, relative_filtered_score]}#, 'mean': [point_score]}
    classifier = linear_model.LogisticRegression()

    encoder = BoWExtractor()
    X_train, X_test, y_train, y_test = load_train_test(name, encoder)
    if TEST:
        X_train = X_train[:TEST_SIZE]
        y_train = y_train[:TEST_SIZE]
        X_test = X_test[:TEST_SIZE]
        y_test = y_test[:TEST_SIZE]

    encoder.clear()
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    predicted_labels = classifier.predict(X_train)
    probabilities = classifier.predict_proba(X_test)[:, 1]
    print(name, X_train.shape[0], len(y_test), 'Trained model. accuracy is', accuracy)
    test_df = pd.DataFrame(columns=['predicted', 'confidence', 'labels', 'accuracy', 'cluster_scoring', 'confidence_model'])

    gmm = GaussianMixture(5, 'tied')
    a = time.time()
    correctness = predicted_labels == y_train
    gmm.fit(X_train[:100].toarray(), correctness[:100])
    predicted_confidence = gmm.predict_proba(X_test[:1].toarray())
    print('gmm', time.time() - a, predicted_confidence)
    df2 = pd.DataFrame(data=np.vstack([probabilities, predicted_confidence, y_test]).T,
                       columns=['predicted', 'confidence', 'labels'])
    df2['accuracy'] = (df2['predicted'].round(0) == df2.labels) * 1.0
    df2['cluster_scoring'] = 'gmm'
    df2['confidence_model'] = 'gmm'
    test_df = test_df.append(df2)


    train_distances = pairwise_distances(X_train)
    test_distances = pairwise_distances(X_test, X_train)
    for method, scoring_functions in methods.items():
        clusters, all_scores = score_clusters(X_train, y_train, predicted_labels, train_distances, method, scoring_functions)

        for i, scores in enumerate(all_scores):
            for confidence_model_name in ['closest_clusters','linear']:
                if confidence_model_name == 'linear':
                    confidence_model = linear_model.Ridge(alpha=0.1)
                    confidence_model.fit(clusters, scores)
                    predicted_confidence = confidence_model.predict(X_test)
                    #scores_dict[scoring_functions[i].__name__] = list(confidence_model.predict(X_train))
                    #conf_items = [(x[0], confidence_model.coef_[x[1]]) for x in encoder.vectorizer.vocabulary_.items()]
                    #confidence_words = sorted(conf_items, key=lambda x: abs(x[1]))[-10:]
                else:
                    predicted_confidence = []
                    for test_index in range(X_test.shape[0]):
                        closest = sorted(range(X_train.shape[0]), key=lambda x: test_distances[test_index, x])[:5]
                        closest_scores = [scores[x] for x in closest]
                        distances = [test_distances[test_index, x] for x in closest]
                        final_score = sum([np.exp(-distance** 3 * 100)*score for score,distance in zip(closest_scores, distances)])
                        predicted_confidence.append(final_score)

                print(name, method, scoring_functions[i].__name__, confidence_model_name,
                      'predicted confidence:', min(predicted_confidence),
                      max(predicted_confidence), 'scores', min(scores), max(scores))
                df2 = pd.DataFrame(data=np.vstack([probabilities, predicted_confidence, y_test]).T,
                                  columns=['predicted', 'confidence', 'labels'])
                df2['accuracy'] = (df2['predicted'].round(0) == df2.labels)*1.0
                df2['cluster_scoring'] = method + '/' + scoring_functions[i].__name__
                df2['confidence_model'] = confidence_model_name
                test_df = test_df.append(df2)


    test_df.to_csv('confidence/'+name+'_test_confidence.csv')
    #pd.DataFrame.from_dict(scores_dict).to_csv('confidence/'+name+'_train_confidence.csv')
    visualize(test_df, name, 1)

def main():
    #names, encoders, strategies, start_round, model = parse_args()
    names = all_datasets[-4:]
    #names = ['mcauley15_balanced_biased', 'mcauley15_balanced', 'mcauley15_biased', 'mcauley15']#, 'pang04', 'pang05', 'pang04_biased', 'pang05_biased', ]
    #names = ['ag_news2']
    for name in names:
        experiment(name)


if __name__ == '__main__':
    #my_test()
    main()
    #runtime_check()
