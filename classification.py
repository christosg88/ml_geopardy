import json
from random import shuffle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tabulate import tabulate


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

    # separate the questions for each category in training set (80%) and test set (20%)
    for i in range(10):
        length = len(quest_per_cat[i])
        train_len = int(length * 0.8)
        train_quest_lst.extend(question for question in quest_per_cat[i][:train_len])
        test_quest_lst.extend(quest_per_cat[i][train_len:])
        train_target_lst.extend([i] * train_len)
        test_target_lst.extend([i] * (length - train_len))

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
                           ("tf_idf_trans", TfidfTransformer())])

    trans_pipe.fit(train_quest_lst)

    train_transformed = trans_pipe.transform(train_quest_lst)
    test_transformed = trans_pipe.transform(test_quest_lst)

    # Naive Bayes classifier for multinomial models
    clf_nb = MultinomialNB()
    clf_nb.fit(train_transformed.toarray(), train_target_lst)
    predicted_nb = clf_nb.predict(test_transformed.toarray())

    # Linear Support Vector Classifier
    clf_svm = LinearSVC()
    clf_svm.fit(train_transformed, train_target_lst)
    predicted_svm = clf_svm.predict(test_transformed)

    results = []

    # accuracy
    accuracy_nb = accuracy_score(test_target_lst, predicted_nb)
    accuracy_svm = accuracy_score(test_target_lst, predicted_svm)
    results.append(["Accuracy", "{:.1f}".format(accuracy_nb * 100), "{:.1f}".format(accuracy_svm * 100)])

    # precision
    precision_nb = precision_score(test_target_lst, predicted_nb, average='macro')
    precision_svm = precision_score(test_target_lst, predicted_svm, average='macro')
    results.append(["Precision", "{:.1f}".format(precision_nb * 100), "{:.1f}".format(precision_svm * 100)])

    # recall
    recall_nb = recall_score(test_target_lst, predicted_nb, average='macro')
    recall_svm = recall_score(test_target_lst, predicted_svm, average='macro')
    results.append(["Recall", "{:.1f}".format(recall_nb * 100), "{:.1f}".format(recall_svm * 100)])

    # F1 score
    f1_score_nb = f1_score(test_target_lst, predicted_nb, average='macro')
    f1_score_svm = f1_score(test_target_lst, predicted_svm, average='macro')
    results.append(["F1 Score", "{:.1f}".format(f1_score_nb * 100), "{:.1f}".format(f1_score_svm * 100)])

    print(tabulate(results, headers=["Metric", "Naive Bayes %", "SVM %"]))
