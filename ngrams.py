# Module for using frequent ngrams for keyphrase combination
# Data comes from http://ngrams.info/
# POS tags: http://ucrel.lancs.ac.uk/claws7tags.html

from collections import Counter
import cPickle as pickle


RAW_BIGRAMS_FILE = 'ngrams/w2c.txt'
RAW_TRIGRAMS_FILE = 'ngrams/w3c.txt'

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

BIGRAM_EXT_TAG_PATTERNS = [
    [ADJECTIVE_TAGS, ADJECTIVE_TAGS],
    [ADJECTIVE_TAGS, EXTENDED_NOUN_TAGS],
    [EXTENDED_NOUN_TAGS, EXTENDED_NOUN_TAGS],
]

TRIGRAM_EXT_TAG_PATTERNS = [
    [ADJECTIVE_TAGS, ADJECTIVE_TAGS, EXTENDED_NOUN_TAGS],
    [ADJECTIVE_TAGS, EXTENDED_NOUN_TAGS, EXTENDED_NOUN_TAGS],
    [EXTENDED_NOUN_TAGS, EXTENDED_NOUN_TAGS, EXTENDED_NOUN_TAGS],
]

BIGRAMS_FILE = "ngrams/bigrams.p"
BIGRAMS_EXT_FILE = "ngrams/bigrams_extended.p"
TRIGRAMS_FILE = "ngrams/trigrams.p"
TRIGRAMS_EXT_FILE = "ngrams/trigrams_extended.p"


def tags_match_pattern(tag_list, patterns_list):
    for pattern in patterns_list:
        if all([tag in tag_set for tag, tag_set in zip(tag_list, pattern)]):
            return True
    return False

def write_pickled_bigrams(extended_nouns=False):
    bigrams = Counter()
    tag_patterns = BIGRAM_EXT_TAG_PATTERNS if extended_nouns else BIGRAM_TAG_PATTERNS
    with open(RAW_BIGRAMS_FILE, 'r') as f:
        for line in f.readlines():
            count, w1, w2, pos1, pos2 = line.strip().split()
            if tags_match_pattern([pos1, pos2], tag_patterns):
                bigrams[(w1, w2)] += int(count)
    write_file = BIGRAMS_EXT_FILE if extended_nouns else BIGRAMS_FILE
    pickle.dump(bigrams, open(write_file, 'wb'))

def write_pickled_trigrams(extended_nouns=False):
    trigrams = Counter()
    tag_patterns = TRIGRAM_EXT_TAG_PATTERNS if extended_nouns else TRIGRAM_TAG_PATTERNS
    with open(RAW_TRIGRAMS_FILE, 'r') as f:
        for line in f.readlines():
            count, w1, w2, w3, pos1, pos2, pos3 = line.strip().split()
            if tags_match_pattern([pos1, pos2, pos3], tag_patterns):
                trigrams[(w1, w2, w3)] += int(count)
    write_file = TRIGRAMS_EXT_FILE if extended_nouns else TRIGRAMS_FILE
    pickle.dump(trigrams, open(write_file, 'wb'))

def read_bigrams(extended_nouns=False):
    read_file = BIGRAMS_EXT_FILE if extended_nouns else BIGRAMS_FILE
    return pickle.load(open(read_file, "rb"))

def read_trigrams(extended_nouns=False):
    read_file = TRIGRAMS_EXT_FILE if extended_nouns else TRIGRAMS_FILE
    return pickle.load(open(read_file, "rb"))

def get_matching_ngrams(ngrams, keyword_set):
    return [
        (words, ngrams[words]) for words in ngrams
        if all([word in keyword_set for word in words])
    ]

if __name__=='__main__':
    # write_pickled_bigrams()
    # write_pickled_trigrams()
    # write_pickled_bigrams(extended_nouns=True)
    # write_pickled_trigrams(extended_nouns=True)
    bigrams = read_bigrams()
    trigrams = read_trigrams()
    print len(bigrams), len(trigrams)
    print get_matching_ngrams(bigrams, set(['health', 'care', 'lawn', 'living', 'room']))
    print get_matching_ngrams(trigrams, set(['global', 'financial', 'crisis', 'economic', 'environmental']))