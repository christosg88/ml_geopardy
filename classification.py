import json
import re
import string
from random import shuffle

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


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

with open('jeopardy.json') as infile:
    data = json.load(infile)

# for question in data:
#     print('Q: {}\nC: {}\n'.format(question['question'], question['category']))

text_clf_nb = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf_svm = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', svm.LinearSVC()),
])

shuffle(data)

questions_per_cat = [[] for i in range(10)]
for question in data:
    questions_per_cat[cat_str2int(question["category"])].append(question["question"])

train_questions_lst = []
train_targets_lst = []
test_questions_lst = []
test_targets_lst = []

for i in range(10):
    length = len(questions_per_cat[i])
    train_len = int(length * 0.8)
    test_len = length - train_len
    train_questions_lst.extend(question for question in questions_per_cat[i][:train_len])
    test_questions_lst.extend(questions_per_cat[i][train_len:])
    train_targets_lst.extend(i for l in range(train_len))
    test_targets_lst.extend(i for l in range(test_len))

text_clf_nb = text_clf_nb.fit(train_questions_lst, train_targets_lst)
predicted_nb = text_clf_nb.predict(test_questions_lst)

text_clf_svm = text_clf_svm.fit(train_questions_lst, train_targets_lst)
predicted_svm = text_clf_svm.predict(test_questions_lst)

# accuracy
accuracy_nb = accuracy_score(test_targets_lst, predicted_nb)
print('Accuracy NB: {}'.format(accuracy_nb))
# accuracy
accuracy_svm = accuracy_score(test_targets_lst, predicted_svm)
print('Accuracy SVM: {}\n'.format(accuracy_svm))

# precision
precision_nb = precision_score(test_targets_lst, predicted_nb, average='macro')
print('Precision NB: {}'.format(precision_nb))
# precision
precision_svm = precision_score(test_targets_lst, predicted_svm, average='macro')
print('Precision SVM: {}\n'.format(precision_svm))

# recall
recall_nb = recall_score(test_targets_lst, predicted_nb, average='macro')
print('Recall NB: {}'.format(recall_nb))
# recall
recall_svm = recall_score(test_targets_lst, predicted_svm, average='macro')
print('Recall SVM: {}\n'.format(recall_svm))

# F1 score
f1_score_nb = f1_score(test_targets_lst, predicted_nb, average='macro')
print('F1 score NB: {}'.format(f1_score_nb))
# F1 score
f1_score_svm = f1_score(test_targets_lst, predicted_svm, average='macro')
print('F1 score SVM: {}'.format(f1_score_svm))
