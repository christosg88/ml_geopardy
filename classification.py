import json
from random import shuffle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


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


with open('jeopardy.json') as infile:
    data = json.load(infile)

# shuffle the data so we get different training and test sets for every run
shuffle(data)

# separate the questions to their categories
quest_per_cat = [[] for i in range(10)]
for quest in data:
    quest_per_cat[cat_str2int(quest["category"])].append(quest["question"])

train_quest_lst = []
train_target_lst = []
test_quest_lst = []
test_target_lst = []

# separate the questions for each category in training set (80%) and test set(20%)
for i in range(10):
    length = len(quest_per_cat[i])
    train_len = int(length * 0.8)
    train_quest_lst.extend(question for question in quest_per_cat[i][:train_len])
    test_quest_lst.extend(quest_per_cat[i][train_len:])
    train_target_lst.extend([i] * train_len)
    test_target_lst.extend([i] * (length - train_len))

# create a vectorizer that will create the vocabulary vectors. Ignore words that appear in only one document
vectorizer = CountVectorizer(min_df=2)
# vocab: A count matrix of type scipy sparse csr, with has the number of occurrences of the j-th word at the i-th
# document at vocab[i][j]
#   [n_samples  x  ~n_features]
vocab_train = vectorizer.fit_transform(train_quest_lst)
vocab_test = vectorizer.transform(test_quest_lst)
# transform the vocabulary from a count matrix to a normalized tf-idf representation. Tf means term-frequency while
# tf-idf means term-frequency times inverse document-frequency. The goal of using tf-idf instead of the raw
# frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very
# frequently in a given corpus and that are hence empirically less informative than features that occur in a small
# fraction of the training corpus.
tf_idf_transformer = TfidfTransformer()
tf_idf_train = tf_idf_transformer.fit_transform(vocab_train)
tf_idf_test = tf_idf_transformer.transform(vocab_test)

# Naive Bayes classifier for multinomial models
clf_nb = MultinomialNB()
clf_nb.fit(tf_idf_train, train_target_lst)
predicted_nb = clf_nb.predict(tf_idf_test)

# Linear Support Vector Classifier
clf_svm = LinearSVC()
clf_svm.fit(tf_idf_train, train_target_lst)
predicted_svm = clf_svm.predict(tf_idf_test)

# accuracy
accuracy_nb = accuracy_score(test_target_lst, predicted_nb)
print('Accuracy NB: {:.1f}%'.format(accuracy_nb * 100))
# accuracy
accuracy_svm = accuracy_score(test_target_lst, predicted_svm)
print('Accuracy SVM: {:.1f}%\n'.format(accuracy_svm * 100))

# precision
precision_nb = precision_score(test_target_lst, predicted_nb, average='macro')
print('Precision NB: {:.1f}%'.format(precision_nb * 100))
# precision
precision_svm = precision_score(test_target_lst, predicted_svm, average='macro')
print('Precision SVM: {:.1f}%\n'.format(precision_svm * 100))

# recall
recall_nb = recall_score(test_target_lst, predicted_nb, average='macro')
print('Recall NB: {:.1f}%'.format(recall_nb * 100))
# recall
recall_svm = recall_score(test_target_lst, predicted_svm, average='macro')
print('Recall SVM: {:.1f}%\n'.format(recall_svm * 100))

# F1 score
f1_score_nb = f1_score(test_target_lst, predicted_nb, average='macro')
print('F1 score NB: {:.1f}%'.format(f1_score_nb * 100))
# F1 score
f1_score_svm = f1_score(test_target_lst, predicted_svm, average='macro')
print('F1 score SVM: {:.1f}%'.format(f1_score_svm * 100))
