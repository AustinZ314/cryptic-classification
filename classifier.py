import string
import pandas as pd
import json
from decrypt.scrape_parse import load_guardian_splits_disjoint

# load data from the disjoint dataset (using disjoint in case later we decide to switch from zero-shot)
_, _, (train, val, test) = load_guardian_splits_disjoint()

# function to grab indicator words from their respective text files
def load_indicators(filename):
    with open(f"indicators/{filename}", "r") as file:
        keywords = set()
        for line in file:
            keywords.add(line.strip())
        return keywords

# keywords for each type of clue from https://www.crosswordunclued.com/2008/09/dictionary.html
# found assemblage keywords at https://www.crosswordtools.com/cryptic-indicators.php
# double definition keywords are the keywords from all the unused categories - classify as double_def if no matches found
indicators = {
    "anagram": load_indicators("anagram.txt"),
    "assemblage": load_indicators("assemblage.txt"),
    "container": load_indicators("container.txt"),
    "hidden_word": load_indicators("hidden_word.txt"),
    "homophone": load_indicators("homophone.txt"),
}

# classify the clue by looking for keywords
def classify_clue(text):

    # normalize clue text (not sure if more should be done?)
    text = text.upper().translate(str.maketrans('', '', string.punctuation))
    words_in_clue = text.split()

    tokens = set()
    n = len(words_in_clue)
    for i in range(n):
        for j in range(i + 1, n + 1):
            phrase = " ".join(words_in_clue[i:j])
            tokens.add(phrase)

    for wordplay_type, keywords in indicators.items():
        if not keywords.isdisjoint(tokens): return wordplay_type

    return "unknown"

# print some outputs for now, just want to see if its working
def print_test():
    for sample in test[:10]:
        category = classify_clue(sample.clue)
        print(f"Clue: {sample.clue} \nPredicted: {category} \nAnswer: {sample.soln_with_spaces}\n\n")

# run the classifier on a certain range of the test set
# save to a csv so it can be opened in sheets for easier manual annotation
def classify_csv(start_ind=0, end_ind=300):
    results = []

    for i, sample in enumerate(test[start_ind:end_ind]):
        category = classify_clue(sample.clue)
        results.append({
            "Index": i + start_ind,
            "Clue": sample.clue,
            "Solution": sample.soln,
            "Predicted_Type": category
        })

    df = pd.DataFrame(results)
    df.to_csv("classifier_predictions.csv", index=False)

# run classifier on every clue in the test set to prep for llm evaluation
def classify_json():
    results = []

    for sample in test:
        category = classify_clue(sample.clue)
        
        clue_obj = sample.__dict__.copy()
        clue_obj["predicted_type"] = category
        results.append(clue_obj)
    
    with open("data/categorized.json", "w") as f:
        json.dump(results, f)

# use one at a time, csv is for testing how good the classifier is and manual annotation
classify_csv()
# classify_json()