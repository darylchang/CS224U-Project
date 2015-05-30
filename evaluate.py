from baseline import *
from parse import handwritten_data_reader, inspec_data_reader, duc_data_reader
import sys

DATASETS = [
    'Handwritten',
    'Inspec',
    'DUC-2001',
]
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

def evaluate_extractor_on_dataset(extractor, dataset):
    reader = READERS[dataset]
    examples = reader()
    tp, fp, fn = 0., 0., 0.
    mistakes = []

    num_correct_r_labels, total_gold_labels = 0, 0
    
    sys.stdout.write('Evaluating extractor on %s dataset' % (dataset))
    for filename, text, gold_labels in examples:
        sys.stdout.write('.')
        sys.stdout.flush()

        num_gold_labels = len(gold_labels)
        extracted_labels = extractor(text, num_gold_labels)
        for extracted_label in extracted_labels:
            if extracted_label in gold_labels:
                tp += 1
            else:
                fp += 1

        missed_labels = gold_labels.difference(extracted_labels)
        fn += len(missed_labels)
        
        correct_labels = gold_labels.intersection(extracted_labels)
        extraneous_labels = set(extracted_labels).difference(gold_labels)
        mistakes.append((filename, correct_labels, missed_labels, extraneous_labels))

        if len(extracted_labels) < num_gold_labels:
            print '\nWARNING: for document %s, extracted %s keyphrases (suggested: %s)' % (filename, len(extracted_labels), num_gold_labels)

        # Update R-Precision stats
        total_gold_labels += num_gold_labels
        num_correct_r_labels += len([
            label for label in extracted_labels[:num_gold_labels]
            if label in gold_labels
        ])
    
    print ' Done.'
    r_precision = float(num_correct_r_labels) / total_gold_labels
    return r_precision, compute_stats(tp, fp, fn), mistakes

def print_results(results):
    print '%-14s%-15s%-13s%-13s%-13s' % ('Dataset', 'R-Precision', 'Precision', 'Recall', 'F1')
    print '\n', '-'*60, '\n'
    for dataset in DATASETS:
        r_precision, precision, recall, f1 = results[dataset]
        print '%-14s%-.3f%-10s%-.3f%-8s%-.3f%-8s%-.3f\n' % (dataset, r_precision, '', precision, '', recall, '', f1)
    print '='*60

def output_mistakes(mistakes_list):
    with open(MISTAKES_FILENAME, 'w') as f:
        for filename, correct_labels, missed_labels, extraneous_labels in mistakes_list:
            f.write('='*79 + '\n')
            f.write('Mistakes for document %s:\n' % (filename))
            f.write('-'*50 + '\n')
            f.write('Correct labels: %s\n\n' % (', '.join(correct_labels)))
            f.write('Missed labels: %s\n\n' % (', '.join(missed_labels)))
            f.write('Extraneous labels: %s\n' % (', '.join(extraneous_labels)))

def evaluate_extractor(extractor):
    results = {}
    mistakes_list = []
    for dataset in DATASETS:
        r_precision, (precision, recall, f1), mistakes = evaluate_extractor_on_dataset(extractor, dataset)
        results[dataset] = (r_precision, precision, recall, f1)
        mistakes_list += mistakes
    print_results(results)
    output_mistakes(mistakes_list)

if __name__=='__main__':
    from degreeCentralityModel import DegreeCentralityModel
    from pageRankModel import PageRankModel
    model = DegreeCentralityModel()
    model2 = PageRankModel()
    model2.evaluate()