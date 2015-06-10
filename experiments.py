import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from parse import *
import re

def count_average_labels():
    inspec_examples = inspec_data_reader()
    token_counts = [len(text.split()) for filename, text, labels in inspec_examples]
    print 'Inspec average number of words: %s' % (np.mean(token_counts))
    duc_examples = duc_data_reader()
    token_counts = [len(text.split()) for filename, text, labels in duc_examples]
    print 'Duc average number of words: %s' % (np.mean(token_counts))

# 98 labels that have punctuation in them, out of 7397 (~1%)
def count_punctuated_keyphrases():
    examples = inspec_data_reader() + duc_data_reader()
    examples_with_labels = []
    tokenizer = RegexpTokenizer(r'[\w\-]+')

    for filename, text, labels in examples:
        print filename
        for label in labels:
            if re.findall('[^\w\s\-/\+\']+', label):
                examples_with_labels.append((label, tokenizer.tokenize(label)))

    for label, tokenized_label in examples_with_labels:
        print label, tokenized_label
    print len(examples_with_labels)

def calc_optimal_R_p():
    dataset_examples = {
        'Inspec': inspec_data_reader(),
        'DUC2001': duc_data_reader(),
    }
    for dataset in dataset_examples:
        label_counts = Counter()
        for filename, text, labels in dataset_examples[dataset]:
            for label in labels:
                if label in text:
                    label_counts['in'] += 1
                else:
                    label_counts['out'] += 1
        total_labels = label_counts['in'] + label_counts['out']
        print '%s dataset: maximum R-precision %.2f' % (dataset, float(label_counts['in']) / total_labels)

def count_keyphrase_lengths():
    examples = inspec_data_reader() + duc_data_reader()
    counter = Counter()
    ngram_in_document = Counter()
    words_in_document = Counter()
    not_in_document = Counter()
    labels_per_document = Counter()
    lengths = []
    doc_lengths = []
    label_counts = []

    for filename, text, labels in examples:
        for label in labels:
            words = label.split()
            num_tokens = len(words)
            lengths.append(num_tokens)
            counter[num_tokens] += 1

            if label in text:
                ngram_in_document[num_tokens] += 1
            elif all([word in text for word in words]):
                words_in_document[num_tokens] += 1
            else:
                not_in_document[num_tokens] += 1

        label_counts.append(len(labels))
        labels_per_document[len(labels)] += 1
        doc_lengths.append(len(text.split()))
    
    for token_length in sorted(counter):
        print '%s tokens: %s keyphrases' % (token_length, counter[token_length])
        print '\tNgram occurred in document: %4s (%.2f%%)' % (ngram_in_document[token_length], 100.0 * ngram_in_document[token_length] / counter[token_length])
        print '\tWords occurred in document: %4s (%.2f%%)' % (words_in_document[token_length], 100.0 * words_in_document[token_length] / counter[token_length])
        print '\tDid not occur in document:  %4s (%.2f%%)' % (not_in_document[token_length], 100.0 * not_in_document[token_length] / counter[token_length])

    print '\n'

    for num_labels in sorted(labels_per_document):
        print '%2s keyphrases: %4s documents' % (num_labels, labels_per_document[num_labels])

    plt.figure(0)
    plt.hist(lengths, bins=9)
    plt.xlabel("Words in keyphrase")
    plt.ylabel("Keyphrase count")

    plt.figure(1)
    plt.hist(label_counts, bins=31)
    plt.xlabel("Number of keyphrases")
    plt.ylabel("Documents")

    plt.figure(2)
    plt.scatter(doc_lengths, label_counts)
    plt.axis([0, 1800, 0, 25])
    plt.xlabel("Document length")
    plt.ylabel("Number of keyphrases")

    plt.show()

if __name__=='__main__':
    #count_keyphrase_lengths()
    # count_punctuated_keyphrases()
    # calc_optimal_R_p()
    count_average_labels()