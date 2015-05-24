from collections import Counter

def baseline_empty_extractor(words):
    return []

def baseline_all_words_extractor(words):
    return words

def baseline_most_frequent_extractor(words, num_keywords=5):
    counter = Counter(words)
    return sorted(counter, key=counter.get, reverse=True)[:num_keywords]