from collections import Counter

def baseline_most_frequent_extractor(words, min_num_labels):
    counter = Counter(words)
    return sorted(counter, key=counter.get, reverse=True)[:min_num_labels]