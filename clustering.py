import json
from time import time

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import f1_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import completeness_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import v_measure_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from tabulate import tabulate
import numpy as np


def cat_str2int(cat):
    if cat == 'SCIENCE & NATURE':
        return 0
    elif cat == 'LITERATURE':
        return 1
    elif cat == 'HISTORY':
        return 2
    elif cat == 'GRAMMAR':
        return 3
    elif cat == 'SPORTS':
        return 4
    elif cat == 'GEOGRAPHY':
        return 5
    elif cat == 'PEOPLE':
        return 6
    elif cat == 'ART':
        return 7
    elif cat == 'FOOD':
        return 8
    elif cat == 'MUSIC':
        return 9


if __name__ == "__main__":
    with open('jeopardy.json') as infile:
        data = json.load(infile)

    # separate the questions to their categories and also merge all questions in a list
    quest_per_cat = [[] for i in range(10)]
    quest_lst = []
    cat_true = []
    for quest in data:
        quest_per_cat[cat_str2int(quest["category"])].append(quest["question"])
        quest_lst.append(quest["question"])
        # quest_lst.append(" ".join([quest["question"], quest["category"]]))
        cat_true.append(cat_str2int(quest["category"]))

    # vectorizer:
    # a vectorizer will create the vocabulary vectors. It will also ignore words that appear in only one document. The
    # vocabulary is a count matrix of type scipy sparse csr, which has the number of occurrences of the j-th word at the
    # i-th document stored at vocab[i][j]
    # tf_idf_trans:
    # transforms the vocabulary from a count matrix to a normalized tf-idf representation. Tf means term-frequency while
    # tf-idf means term-frequency times inverse document-frequency. The goal of using tf-idf instead of the raw
    # frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very
    # frequently in a given corpus and that are hence empirically less informative than features that occur in a small
    # fraction of the training corpus.
    trans_pipe = Pipeline([("vectorizer", CountVectorizer(min_df=2)),
                           ("tf_idf_trans", TfidfTransformer()),
                           ("normalizer", Normalizer(copy=False))])

    # transform the whole dataset and fit the transformation pipe to it
    t0 = time()
    trans_pipe.fit(quest_lst)
    train_transformed = trans_pipe.transform(quest_lst)
    t1 = time()
    print("Data transformed in {:.2f}s".format(t1 - t0))

    # find the cluster center for each category to help KMeans
    print("\nFinding cluster center per category.")
    cluster_center_per_cat = np.empty((0, train_transformed.shape[1]))
    for cat in range(10):
        print("Category {}".format(cat))
        transformed = trans_pipe.transform(quest_per_cat[cat])
        clust_km = KMeans(n_clusters=1, n_init=20, n_jobs=-1)
        clust_km.fit(transformed)
        cluster_center_per_cat = np.append(cluster_center_per_cat, clust_km.cluster_centers_, axis=0)

    clust_km = KMeans(n_clusters=10, n_init=1, init=cluster_center_per_cat)
    cat_km = clust_km.fit_predict(train_transformed)
    t2 = time()
    print("\nKMeans fit the data and predicted the categories in {:.2f}s".format(t2 - t1))

    results = []
    # F1 Score
    f1_score_km = f1_score(cat_km, cat_true, average="micro")
    results.append(["F1 Score", "{:.1f}".format(f1_score_km * 100)])

    # adjusted rand score
    adjusted_rand_score_km = adjusted_rand_score(cat_km, cat_true)
    results.append(["Adjusted Rand Score", "{:.1f}".format(adjusted_rand_score_km * 100)])

    # adjusted mutual info score
    adjusted_mutual_info_score_km = adjusted_mutual_info_score(cat_km, cat_true)
    results.append(["Adjusted mutual info score", "{:.1f}".format(adjusted_mutual_info_score_km * 100)])

    # homogeneity score
    homogeneity_score_km = homogeneity_score(cat_km, cat_true)
    results.append(["Homogeneity score", "{:.1f}".format(homogeneity_score_km * 100)])

    # completeness score
    completeness_score_km = completeness_score(cat_km, cat_true)
    results.append(["Completeness score", "{:.1f}".format(completeness_score_km * 100)])

    # v measure score
    v_measure_score_km = v_measure_score(cat_km, cat_true)
    results.append(["V measure score", "{:.1f}".format(v_measure_score_km * 100)])

    print()
    print(tabulate(results, headers=["Metric", "KMeans %"]))
