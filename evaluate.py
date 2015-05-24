from baseline import *
from parse import handwritten_data_reader, inspec_data_reader

DATASETS = ['Handwritten', 'Inspec']#, 'DUC-2001']
READERS = {
    'Handwritten': handwritten_data_reader,
    'Inspec': inspec_data_reader,
}


def F1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)

def compute_stats(tp, fp, fn):
    precision = 0. if tp + fp == 0 else tp / (tp + fp)
    recall = 0. if tp + fn == 0 else tp / (tp + fn)
    return (precision, recall, F1(precision, recall))

def evaluate_extractor_on_reader(extractor, reader, **kwargs):
    examples = reader(**kwargs)
    tp, fp, fn = 0., 0., 0.
    for tokens, labels in examples:
        extracted_labels = extractor(tokens)
        for extracted_label in extracted_labels:
            if extracted_label in labels:
                tp += 1
            else:
                fp += 1
        fn += len(labels.difference(extracted_labels))
    return compute_stats(tp, fp, fn)

def print_results(results):
    print '%-14s%-12s%-12s%-12s' % ('Dataset', 'Precision', 'Recall', 'F1')
    print '------------------------------------------'
    for dataset in DATASETS:
        precision, recall, f1 = results[dataset]
        print '%-14s%-.2f%-8s%-.2f%-8s%-.2f\n' % (dataset, precision, '', recall, '', f1)
    print '=========================================='

def evaluate_extractor(extractor, **kwargs):
    results = {}
    for dataset in DATASETS:
        results[dataset] = evaluate_extractor_on_reader(extractor, READERS[dataset], **kwargs)
    print_results(results)

if __name__=='__main__':
    evaluate_extractor(baseline_all_words_extractor)