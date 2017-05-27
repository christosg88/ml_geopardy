import json
import string
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *


def remove_special_chars(s):
    # replace more than one characters
    s = s.replace('\'s', '').replace('-', ' ')

    # remove all punctuation
    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    translation = str.maketrans('', '', string.punctuation)

    return s.translate(translation)


includes_number_regex = re.compile(r'\d+')
html_tags_regex = re.compile(r'<.+>')

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

init_quest_cnt = 0
init_cat = Counter()

for question in data:
    init_quest_cnt += 1
    init_cat.update([question["category"]])

print("# QUESTIONS IN INITIAL DATASET: {}\n".format(init_quest_cnt))
print("# CATEGORIES IN INITIAL DATASET: {}\n".format(len(init_cat)))
print("MOST COMMON CATEGORIES WITH NUMBER OF QUESTIONS:")
for cat, num_quest in init_cat.most_common(10):
    print("Category: {}\nQuestions: {}\n".format(cat, num_quest))
print("================================================================================")

stemmer = PorterStemmer()
stopWords = stopwords.words('english')

quest_lst = []
words_and_freq = Counter()
for question in data:
    if question["category"] in category_dict:
        q = question["question"]
        q = q.lower()
        q = html_tags_regex.sub('', q)
        q = remove_special_chars(q)
        words_lst = nltk.word_tokenize(q)

        words_lst = [stemmer.stem(word)
                     for word in words_lst
                     if word not in stopWords and not includes_number_regex.match(word)]

        words_and_freq.update(words_lst)

        quest_lst.append({"question": ' '.join(words_lst), "category": category_dict[question["category"]]})

trimmed_cat = Counter(quest["category"] for quest in quest_lst)

print("# QUESTIONS IN TRIMMED DATASET: {}\n".format(len(quest_lst)))
print("# CATEGORIES IN TRIMMED DATASET: {}\n".format(10))
print("MOST COMMON CATEGORIES WITH NUMBER OF QUESTIONS:")
for cat, num_quest in trimmed_cat.most_common(10):
    print("Category: {}\nQuestions: {}\n".format(cat, num_quest))
print("================================================================================")

print("# WORDS IN TRIMMED DATASET: {}\n".format(len(words_and_freq)))

with open('jeopardy.json', 'w') as outfile:
    json.dump(quest_lst, outfile)
