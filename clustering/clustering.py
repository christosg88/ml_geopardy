import json
import numpy as np
import string
from random import shuffle
import re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import fowlkes_mallows_score

from time import time

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


def cat_int2str(cat):
    if cat == 0:
        return 'SCIENCE & NATURE'
    elif cat == 1:
        return 'LITERATURE'
    elif cat == 2:
        return 'HISTORY'
    elif cat == 3:
        return 'GRAMMAR'
    elif cat == 4:
        return 'SPORTS'
    elif cat == 5:
        return 'GEOGRAPHY'
    elif cat == 6:
        return 'PEOPLE'
    elif cat == 7:
        return 'ART'
    elif cat == 8:
        return 'FOOD'
    elif cat == 9:
        return 'MUSIC'


def remove_special_chars(s):
    # replace more than one characters
    s = s.replace('\'s', '').replace('-', ' ')

    # remove all punctuation
    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    translation = str.maketrans('', '', string.punctuation)

    return s.translate(translation)


number_regex = re.compile(r'^\d+$')
html_tags_regex = re.compile(r'<.+>')

with open('../jeopardy.json') as infile:
    data = json.load(infile)

# for question in data:
#     print('Q: {}\nC: {}\n'.format(question['question'], question['category']))

question_lst = []
labels_true = []

for question in data:
	question_lst.append(question['question'])
	labels_true.append(cat_str2int(question['category']))

vectorizer = TfidfVectorizer(use_idf = True)
X =  vectorizer.fit_transform(question_lst)
print("n_samples: %d, n_features: %d" % X.shape)

text_clust_kmeans = KMeans(n_clusters = 10, random_state = 1,n_init = 5,max_iter = 100).fit(X)
labels_pred_kmeans = text_clust_kmeans.labels_ 

order_centroids = text_clust_kmeans.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(10):
    print("Cluster %d:" % i, end="")
    for ind in order_centroids[i, :10]:
        print(" %s" % terms[ind], end="")
    print()

# adjusted rand score
adjusted_rand_score_kmeans = adjusted_rand_score(labels_pred_kmeans, labels_true)
print('Adjusted Rand Score KMeans: {}'.format(adjusted_rand_score_kmeans))

# adjusted mutual info score
adjusted_mutual_info_score_kmeans = adjusted_mutual_info_score(labels_pred_kmeans, labels_true)
print('Adjusted mutual info score KMeans: {}\n'.format(adjusted_mutual_info_score_kmeans))

# homogeneity score
homogeneity_score_kmeans = homogeneity_score(labels_pred_kmeans, labels_true)
print('Homogeneity score KMeans: {}\n'.format(homogeneity_score_kmeans))

# completeness score
completeness_score_kmeans = completeness_score(labels_pred_kmeans, labels_true)
print('Completeness score KMeans: {}\n'.format(completeness_score_kmeans))

# v measure score
v_measure_score_kmeans = v_measure_score(labels_pred_kmeans, labels_true)
print('V measure score KMeans: {}\n'.format(v_measure_score_kmeans))

# fowlkes mallows score
fowlkes_mallows_score_kmeans = fowlkes_mallows_score(labels_pred_kmeans, labels_true)
print('Fowlkes-Mallows score KMeans: {}\n'.format(fowlkes_mallows_score_kmeans))
