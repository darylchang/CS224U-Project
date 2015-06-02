# Module for using frequent ngrams for keyphrase combination
# Data comes from http://ngrams.info/
# POS tags: http://ucrel.lancs.ac.uk/claws7tags.html

from collections import Counter


BIGRAM_FILE = 'ngrams/w2c.txt'
TRIGRAM_FILE = 'ngrams/w3c.txt'

ADJECTIVE_TAGS = set(['jj', 'jjr', 'jjt'])
BASIC_NOUN_TAGS = set(['nn', 'nn1', 'nn2'])
EXTENDED_NOUN_TAGS = set(['nd1', 'nn', 'nn1', 'nn2', 'nna', 'nnb', 'nnl1', 'nnl2', 'nno', 'nno2', 'nnt1', 'nnt2', 'nnu', 'nnu1', 'nnu2', 'np', 'np1', 'np2', 'npd1', 'npd2', 'npm1', 'npm2'])

BIGRAM_TAG_PATTERNS = [
    [ADJECTIVE_TAGS, ADJECTIVE_TAGS],
    [ADJECTIVE_TAGS, BASIC_NOUN_TAGS],
    [BASIC_NOUN_TAGS, BASIC_NOUN_TAGS],
]

TRIGRAM_TAG_PATTERNS = [
    [ADJECTIVE_TAGS, ADJECTIVE_TAGS, BASIC_NOUN_TAGS],
    [ADJECTIVE_TAGS, BASIC_NOUN_TAGS, BASIC_NOUN_TAGS],
    [BASIC_NOUN_TAGS, BASIC_NOUN_TAGS, BASIC_NOUN_TAGS],
]


def tags_match_pattern(tag_list, patterns_list):
    for pattern in patterns_list:
        if all([tag in tag_set for tag, tag_set in zip(tag_list, pattern)]):
            return True
    return False

def read_bigrams():
    bigrams = Counter()
    with open(BIGRAM_FILE, 'r') as f:
        for line in f.readlines():
            count, w1, w2, pos1, pos2 = line.strip().split()
            if tags_match_pattern([pos1, pos2], BIGRAM_TAG_PATTERNS):
                bigrams[(w1, w2)] += int(count)
    return bigrams

def read_trigrams():
    trigrams = Counter()
    with open(TRIGRAM_FILE, 'r') as f:
        for line in f.readlines():
            count, w1, w2, w3, pos1, pos2, pos3 = line.strip().split()
            if tags_match_pattern([pos1, pos2, pos3], TRIGRAM_TAG_PATTERNS):
                trigrams[(w1, w2, w3)] += int(count)
    return trigrams

def get_matching_ngrams(ngrams, keyword_set):
    return [
        (words, ngrams[words]) for words in ngrams
        if all([word in keyword_set for word in words])
    ]

if __name__=='__main__':
    bigrams = read_bigrams()
    trigrams = read_trigrams()
    print len(bigrams), len(trigrams)
    print get_matching_ngrams(bigrams, set(['health', 'care', 'lawn', 'living', 'room']))
    print get_matching_ngrams(trigrams, set(['global', 'financial', 'crisis', 'economic', 'environmental']))