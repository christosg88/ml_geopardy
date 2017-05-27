import json

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import completeness_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import v_measure_score


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


with open('jeopardy.json') as infile:
    data = json.load(infile)

quest_lst = []
cat_true = []
for quest in data:
    quest_lst.append(quest["question"])
    cat_true.append(cat_str2int(quest["category"]))

# create a vectorizer that will create the vocabulary vectors. Ignore words that appear in only one document
vectorizer = CountVectorizer(min_df=2)
# vocab: A count matrix of type scipy sparse csr, with has the number of occurrences of the j-th word at the i-th
# document at vocab[i][j]
#   [n_samples  x  ~n_features]
vocab = vectorizer.fit_transform(quest_lst)
# transform the vocabulary from a count matrix to a normalized tf-idf representation. Tf means term-frequency while
# tf-idf means term-frequency times inverse document-frequency. The goal of using tf-idf instead of the raw
# frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very
# frequently in a given corpus and that are hence empirically less informative than features that occur in a small
# fraction of the training corpus.
tf_idf_transformer = TfidfTransformer()
tf_idf = tf_idf_transformer.fit_transform(vocab)

# print("[!] Performing dimensionality reduction using LSA")
# # Vectorizer results are normalized, which makes KMeans behave as spherical k-means for better results. Since LSA/SVD
# #  results are not normalized, we have to redo the normalization.
# svd = TruncatedSVD(200)
# normalizer = Normalizer(copy=False)
# lsa = make_pipeline(svd, normalizer)
#
# X = lsa.fit_transform(X)
# t3 = time()
# print("Reduced dimensions in {:.2f}s".format(t3 - t2))

km = KMeans(n_clusters=2, n_jobs=1)
km.fit(X)
#
# em = GaussianMixture(n_components=10)
# em.fit(X.toarray())

labels_pred_km = km.predict(X)
# labels_pred_em = em.predict(X)

# original_space_centroids = svd.inverse_transform(km.cluster_centers_)
# order_centroids = original_space_centroids.argsort()[:, ::-1]
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms = vec.get_feature_names()
for i in range(2):
    print("Cluster %d:" % i, end="")
    for ind in order_centroids[i, :10]:
        print(" %s" % terms[ind], end="")
    print()
print()

# adjusted rand score
adjusted_rand_score_km = adjusted_rand_score(labels_pred_km, labels_true)
print('Adjusted Rand Score KMeans: {}'.format(adjusted_rand_score_km))

# adjusted rand score
# adjusted_rand_score_em = adjusted_rand_score(labels_pred_em, labels_true)
# print('Adjusted Rand Score Expectation Maximization: {}\n'.format(adjusted_rand_score_em))

# adjusted mutual info score
adjusted_mutual_info_score_km = adjusted_mutual_info_score(labels_pred_km, labels_true)
print('Adjusted mutual info score KMeans: {}'.format(adjusted_mutual_info_score_km))

# adjusted mutual info score
# adjusted_mutual_info_score_em = adjusted_mutual_info_score(labels_pred_em, labels_true)
# print('Adjusted mutual info score Expectation Maximization: {}\n'.format(adjusted_mutual_info_score_em))

# homogeneity score
homogeneity_score_km = homogeneity_score(labels_pred_km, labels_true)
print('Homogeneity score KMeans: {}'.format(homogeneity_score_km))

# homogeneity score
# homogeneity_score_em = homogeneity_score(labels_pred_em, labels_true)
# print('Homogeneity score Expectation Maximization: {}\n'.format(homogeneity_score_em))

# completeness score
completeness_score_km = completeness_score(labels_pred_km, labels_true)
print('Completeness score KMeans: {}'.format(completeness_score_km))

# completeness score
# completeness_score_em = completeness_score(labels_pred_em, labels_true)
# print('Completeness score Expectation Maximization: {}\n'.format(completeness_score_em))

# v measure score
v_measure_score_km = v_measure_score(labels_pred_km, labels_true)
print('V measure score KMeans: {}'.format(v_measure_score_km))

# v measure score
# v_measure_score_em = v_measure_score(labels_pred_em, labels_true)
# print('V measure score Expectation Maximization: {}\n'.format(v_measure_score_em))
