import json
import nltk
import numpy as np
import pprint
import re
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import *
from pprint import pprint

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def cat_str2int(cat):
    if cat == 'SCIENCE & NATURE':   return 1
    elif cat == 'LITERATURE':       return 2
    elif cat == 'HISTORY':          return 3
    elif cat == 'GRAMMAR':          return 4
    elif cat == 'SPORTS':           return 5
    elif cat == 'GEOGRAPHY':        return 6
    elif cat == 'PEOPLE':           return 7
    elif cat == 'ART':              return 8
    elif cat == 'FOOD':             return 9
    elif cat == 'MUSIC':            return 10

def cat_int2str(cat):
    if cat == 1:     return 'SCIENCE & NATURE'
    elif cat == 2:   return 'LITERATURE'
    elif cat == 3:   return 'HISTORY'
    elif cat == 4:   return 'GRAMMAR'
    elif cat == 5:   return 'SPORTS'
    elif cat == 6:   return 'GEOGRAPHY'
    elif cat == 7:   return 'PEOPLE'
    elif cat == 8:   return 'ART'
    elif cat == 9:   return 'FOOD'
    elif cat == 10:  return 'MUSIC'

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

text_clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

questions_lst = [question['question'] for question in data]
train_questions_lst = questions_lst[:15000]
test_questions_lst = questions_lst[15001:]
targets_lst = list(map(cat_str2int, [question['category'] for question in data]))
train_targets_lst = targets_lst[:15000]
test_targets_lst = targets_lst[15001:]

text_clf = text_clf.fit(train_questions_lst, train_targets_lst)

predicted = text_clf.predict(test_questions_lst)

print(np.mean(predicted == test_targets_lst))

#accuracy
accuracy = accuracy_score(test_targets_lst,predicted)
print(['Accuracy: ' + str(accuracy)])

#precision
precision = precision_score(test_targets_lst,predicted,average = 'macro')
print(['Precision: ' + str(precision)])

#recall
recall = recall_score(test_targets_lst,predicted,average = 'macro')
print(['Recall: ' + str(recall)])

#F1 score
f1_score = f1_score(test_targets_lst,predicted,average = 'macro')
print(['F1 score: ' + str(f1_score)])


# stemmer = PorterStemmer()
# stopWords = set(stopwords.words('english'))

# edited_lst = []
# new_question_lst = ['the painting was really beautiful and the artist was young',
#                     'this country\'s citizens are really wealthy']
# for q in new_question_lst:
#     q = q.lower()
#     q = html_tags_regex.sub('', q)
#     q = remove_special_chars(q)
#     words_lst = nltk.word_tokenize(q)
#     words_lst = [stemmer.stem(word) for word in words_lst if (not number_regex.match(word) and word not in stopWords)]
#     q = ' '.join(words_lst)
#     edited_lst.append(q)

# predicted = text_clf.predict(edited_lst)

# for doc, category in zip(edited_lst, predicted):
#     print('{} => {}'.format(doc, cat_int2str(category)))

