from baseline import *
from parse import handwritten_data_reader, inspec_data_reader, duc_data_reader


DATASETS = ['Handwritten', 'Inspec', 'DUC-2001']
READERS = {
    'Handwritten': handwritten_data_reader,
    'Inspec': inspec_data_reader,
    'DUC-2001': duc_data_reader,
}
MISTAKES_FILENAME = 'mistakes.txt'


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
    mistakes = []
    
    for filename, tokens, labels in examples:
        extracted_labels = extractor(tokens)
        for extracted_label in extracted_labels:
            if extracted_label in labels:
                tp += 1
            else:
                fp += 1
        
        missed_labels = labels.difference(extracted_labels)
        fn += len(missed_labels)
        
        correct_labels = labels.intersection(extracted_labels)
        extraneous_labels = set(extracted_labels).difference(labels)
        mistakes.append((filename, correct_labels, missed_labels, extraneous_labels))
        
    return compute_stats(tp, fp, fn), mistakes

def print_results(results):
    print '%-14s%-12s%-12s%-12s' % ('Dataset', 'Precision', 'Recall', 'F1')
    print '-------------------------------------------\n'
    for dataset in DATASETS:
        precision, recall, f1 = results[dataset]
        print '%-14s%-.3f%-7s%-.3f%-7s%-.3f\n' % (dataset, precision, '', recall, '', f1)
    print '==========================================='

def output_mistakes(mistakes_list):
    with open(MISTAKES_FILENAME, 'w') as f:
        for filename, correct_labels, missed_labels, extraneous_labels in mistakes_list:
            f.write('='*79 + '\n')
            f.write('Mistakes for document %s:\n' % (filename))
            f.write('-'*50 + '\n')
            f.write('Correct labels: %s\n\n' % (', '.join(correct_labels)))
            f.write('Missed labels: %s\n\n' % (', '.join(missed_labels)))
            f.write('Extraneous labels: %s\n' % (', '.join(extraneous_labels)))

def evaluate_extractor(extractor, **kwargs):
    results = {}
    mistakes_list = []
    for dataset in DATASETS:
        reader_results, mistakes = evaluate_extractor_on_reader(extractor, READERS[dataset], **kwargs)
        results[dataset] = reader_results
        mistakes_list += mistakes
    print_results(results)
    output_mistakes(mistakes_list)

if __name__=='__main__':
    evaluate_extractor(baseline_most_frequent_extractor)