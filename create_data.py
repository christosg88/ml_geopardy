import json
from pprint import pprint
from collections import Counter

with open('JEOPARDY_QUESTIONS1.json') as infile:
    data = json.load(infile)

category_dict = {
    "ANATOMY":              "SCIENCE & NATURE",
    "ANIMALS":              "SCIENCE & NATURE",
    "ASTRONOMY":            "SCIENCE & NATURE",
    "BIOLOGY":              "SCIENCE & NATURE",
    "CHEMISTRY":            "SCIENCE & NATURE",
    "GENERAL SCIENCE":      "SCIENCE & NATURE",
    "MEDICINE":             "SCIENCE & NATURE",
    "NATURE":               "SCIENCE & NATURE",
    "PHYSICS":              "SCIENCE & NATURE",
    "SCIENCE & NATURE":     "SCIENCE & NATURE",
    "SCIENCE":              "SCIENCE & NATURE",
    "THE BODY HUMAN":       "SCIENCE & NATURE",
    "WEIGHTS & MEASURES":   "SCIENCE & NATURE",
    "ZOOLOGY":              "SCIENCE & NATURE",

    "AMERICAN LITERATURE":  "LITERATURE",
    "AUTHORS":              "LITERATURE",
    "BOOKS & AUTHORS":      "LITERATURE",
    "FICTIONAL CHARACTERS": "LITERATURE",
    "LITERATURE":           "LITERATURE",
    "POETS & POETRY":       "LITERATURE",
    "SHAKESPEARE":          "LITERATURE",

    "AMERICAN HISTORY":         "HISTORY",
    "EUROPEAN HISTORY":         "HISTORY",
    "HISTORY":                  "HISTORY",
    "THE AMERICAN REVOLUTION":  "HISTORY",
    "THE CIVIL WAR":            "HISTORY",
    "U.S. HISTORY":             "HISTORY",
    "WORLD HISTORY":            "HISTORY",

    "10-LETTER WORDS":          "GRAMMAR",
    "3-LETTER WORDS":           "GRAMMAR",
    "4-LETTER WORDS":           "GRAMMAR",
    "5-LETTER WORDS":           "GRAMMAR",
    "FOREIGN WORDS & PHRASES":  "GRAMMAR",
    "IN THE DICTIONARY":        "GRAMMAR",
    "LANGUAGES":                "GRAMMAR",
    "VOCABULARY":               "GRAMMAR",
    "WORD ORIGINS":             "GRAMMAR",

    "SPORTS":                   "SPORTS",

    "AROUND THE WORLD": "GEOGRAPHY",
    "BODIES OF WATER":  "GEOGRAPHY",
    "GEOGRAPHY":        "GEOGRAPHY",
    "ISLANDS":          "GEOGRAPHY",
    "LAKES & RIVERS":   "GEOGRAPHY",
    "MOUNTAINS":        "GEOGRAPHY",
    "STATE CAPITALS":   "GEOGRAPHY",
    "U.S. CITIES":      "GEOGRAPHY",
    "U.S. GEOGRAPHY":   "GEOGRAPHY",
    "WORLD CAPITALS":   "GEOGRAPHY",
    "WORLD CITIES":     "GEOGRAPHY",
    "WORLD GEOGRAPHY":  "GEOGRAPHY",

    "ACTORS & ACTRESSES":   "PEOPLE",
    "ARTISTS":              "PEOPLE",
    "EXPLORERS":            "PEOPLE",
    "FAMOUS AMERICANS":     "PEOPLE",
    "FAMOUS NAMES":         "PEOPLE",
    "FIRST LADIES":         "PEOPLE",
    "HISTORIC NAMES":       "PEOPLE",
    "NOTABLE NAMES":        "PEOPLE",
    "PEOPLE":               "PEOPLE",
    "QUOTATIONS":           "PEOPLE",
    "SCIENTISTS":           "PEOPLE",
    "U.S. PRESIDENTS":      "PEOPLE",
    "WORLD LEADERS":        "PEOPLE",

    "ART & ARTISTS":    "ART",
    "ART":              "ART",
    "BALLET":           "ART",
    "NONFICTION":       "ART",
    "THE MOVIES":       "ART",
    "THEATRE":          "ART",

    "FOOD & DRINK":         "FOOD",
    "FOOD":                 "FOOD",
    "FRUITS & VEGETABLES":  "FOOD",

    "CLASSICAL MUSIC":      "MUSIC",
    "COMPOSERS":            "MUSIC",
    "MUSIC":                "MUSIC",
    "MUSICAL INSTRUMENTS":  "MUSIC",
    "OPERA":                "MUSIC",
    "POP MUSIC":            "MUSIC"
}

count = 0
jeopard = []
for q in data:
    if q["category"] in category_dict:
        q["category"] = category_dict[q["category"]]
        jeopard.append(q)
        count += 1

pprint(count)

categories = []
for q in data:
    categories.append(q["category"])

with open('jeopardy.json', 'w') as outfile:
    json.dump(jeopard, outfile)

pprint(jeopard)

# pprint(Counter(categories).most_common(10))
