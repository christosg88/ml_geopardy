import json
import string
import re
import nltk
from pprint import pprint
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import *

number_regex = re.compile(r'^\d+$')
html_tags_regex = re.compile(r'<.+>')


def remove_special_chars(s):
    # replace more than one characters
    s = s.replace('\'s', '').replace('-', ' ')

    # remove all punctuation
    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    translation = str.maketrans('', '', string.punctuation)

    return s.translate(translation)


with open('JEOPARDY_QUESTIONS1.json') as infile:
    data = json.load(infile)

category_dict = {
    "ANATOMY": "SCIENCE & NATURE",
    "ANIMALS": "SCIENCE & NATURE",
    "ASTRONOMY": "SCIENCE & NATURE",
    "BIOLOGY": "SCIENCE & NATURE",
    "CHEMISTRY": "SCIENCE & NATURE",
    "GENERAL SCIENCE": "SCIENCE & NATURE",
    "MEDICINE": "SCIENCE & NATURE",
    "NATURE": "SCIENCE & NATURE",
    "PHYSICS": "SCIENCE & NATURE",
    "SCIENCE & NATURE": "SCIENCE & NATURE",
    "SCIENCE": "SCIENCE & NATURE",
    "THE BODY HUMAN": "SCIENCE & NATURE",
    "WEIGHTS & MEASURES": "SCIENCE & NATURE",
    "ZOOLOGY": "SCIENCE & NATURE",

    "AMERICAN LITERATURE": "LITERATURE",
    "AUTHORS": "LITERATURE",
    "BOOKS & AUTHORS": "LITERATURE",
    "FICTIONAL CHARACTERS": "LITERATURE",
    "LITERATURE": "LITERATURE",
    "POETS & POETRY": "LITERATURE",
    "SHAKESPEARE": "LITERATURE",

    "AMERICAN HISTORY": "HISTORY",
    "EUROPEAN HISTORY": "HISTORY",
    "HISTORY": "HISTORY",
    "THE AMERICAN REVOLUTION": "HISTORY",
    "THE CIVIL WAR": "HISTORY",
    "U.S. HISTORY": "HISTORY",
    "WORLD HISTORY": "HISTORY",

    "10-LETTER WORDS": "GRAMMAR",
    "3-LETTER WORDS": "GRAMMAR",
    "4-LETTER WORDS": "GRAMMAR",
    "5-LETTER WORDS": "GRAMMAR",
    "FOREIGN WORDS & PHRASES": "GRAMMAR",
    "IN THE DICTIONARY": "GRAMMAR",
    "LANGUAGES": "GRAMMAR",
    "VOCABULARY": "GRAMMAR",
    "WORD ORIGINS": "GRAMMAR",

    "SPORTS": "SPORTS",

    "AROUND THE WORLD": "GEOGRAPHY",
    "BODIES OF WATER": "GEOGRAPHY",
    "GEOGRAPHY": "GEOGRAPHY",
    "ISLANDS": "GEOGRAPHY",
    "LAKES & RIVERS": "GEOGRAPHY",
    "MOUNTAINS": "GEOGRAPHY",
    "STATE CAPITALS": "GEOGRAPHY",
    "U.S. CITIES": "GEOGRAPHY",
    "U.S. GEOGRAPHY": "GEOGRAPHY",
    "WORLD CAPITALS": "GEOGRAPHY",
    "WORLD CITIES": "GEOGRAPHY",
    "WORLD GEOGRAPHY": "GEOGRAPHY",

    "ACTORS & ACTRESSES": "PEOPLE",
    "ARTISTS": "PEOPLE",
    "EXPLORERS": "PEOPLE",
    "FAMOUS AMERICANS": "PEOPLE",
    "FAMOUS NAMES": "PEOPLE",
    "FIRST LADIES": "PEOPLE",
    "HISTORIC NAMES": "PEOPLE",
    "NOTABLE NAMES": "PEOPLE",
    "PEOPLE": "PEOPLE",
    "QUOTATIONS": "PEOPLE",
    "SCIENTISTS": "PEOPLE",
    "U.S. PRESIDENTS": "PEOPLE",
    "WORLD LEADERS": "PEOPLE",

    "ART & ARTISTS": "ART",
    "ART": "ART",
    "BALLET": "ART",
    "NONFICTION": "ART",
    "THE MOVIES": "ART",
    "THEATRE": "ART",

    "FOOD & DRINK": "FOOD",
    "FOOD": "FOOD",
    "FRUITS & VEGETABLES": "FOOD",

    "CLASSICAL MUSIC": "MUSIC",
    "COMPOSERS": "MUSIC",
    "MUSIC": "MUSIC",
    "MUSICAL INSTRUMENTS": "MUSIC",
    "OPERA": "MUSIC",
    "POP MUSIC": "MUSIC"
}

initial_dataset_cnt = 0;
initial_categories_set = set()

for question in data:
    initial_dataset_cnt += 1
    initial_categories_set.add(question["category"])

print("# questions in initial dataset: {}".format(initial_dataset_cnt))
print("# categories in initial dataset: {}".format(len(initial_categories_set)))

stemmer = PorterStemmer()
stopWords = set(stopwords.words('english'))

dataset_cnt = 0
questions_lst = []
for question in data:
    if question["category"] in category_dict:
        q = question["question"]
        q = q.lower()
        q = html_tags_regex.sub('', q)
        q = remove_special_chars(q)
        words_lst = nltk.word_tokenize(q)
        words_lst = [stemmer.stem(word) for word in words_lst if
                     (not number_regex.match(word) and word not in stopWords)]
        question["question"] = ' '.join(words_lst)

        question["category"] = category_dict[question["category"]]

        questions_lst.append(question)
        dataset_cnt += 1

print("# questions in trimmed dataset: {}".format(dataset_cnt))
print("# categories in trimmed dataset: {}".format(10))

questions_per_category = Counter(question["category"] for question in questions_lst)
print("# questions per category:")
pprint(questions_per_category)

words_and_freq = Counter()
for question in questions_lst:
    print('Q: {}\nA: {}\n'.format(question['question'], question['category']))
    words_and_freq.update(Counter(question['question'].split(' ')))

print("# words: {}".format(len(words_and_freq)))
# pprint(words_and_freq)

with open('jeopardy.json', 'w') as outfile:
    json.dump(questions_lst, outfile)
