from decrypt.scrape_parse import load_guardian_splits_disjoint

# load data from the disjoint dataset (using disjoint in case later we decide to switch from zero-shot)
_, _, (train, val, test) = load_guardian_splits_disjoint()

# keywords for each type of clue from https://www.crosswordunclued.com/2008/09/dictionary.html
# might be faster to have the values be sets instead of lists, using set intersection or similar method might be more efficient
indicators = {
    "anagram": ["", ],
    "assemblage": ["", ],
    "container": ["", ],
    "hidden_word": ["", ],
    "double_def": ["", ], 
}

# classify the clue by looking for keywords
def classify_clue(text):

    # normalize clue text (not sure if more should be done?)
    text = text.lower()

    for wordplay_type, keywords in indicators.items():
        if any(word in text for word in keywords):
            return wordplay_type
    return "" # need some sort of catch all or placeholder, unsure

# print some outputs for now, just want to see if its working
for sample in test[:10]:
    category = classify_clue(sample.clue_text)
    print(f"Clue: {sample.clue_text} \tPredicted: {category}")