import string
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
    "double_def": load_indicators("double_def.txt"), 
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
        if wordplay_type == "double_def": # ensure that double_def is processed last
            continue
        else:
            if not keywords.isdisjoint(tokens): return wordplay_type
    
    double_def_keywords = indicators.get("double_def")
    if double_def_keywords.isdisjoint(tokens):
        return "double_def"

    return "unknown"

# print some outputs for now, just want to see if its working
for sample in test[:10]:
    category = classify_clue(sample.clue)
    print(f"Clue: {sample.clue} \nPredicted: {category} \nAnswer: {sample.soln_with_spaces}\n\n")