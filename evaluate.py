from baseline import *
from parse import handwritten_data_reader, inspec_data_reader, duc_data_reader
from pattern.en import singularize
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

def contains_sublist(lst, sublst):
    n = len(sublst)
    return any((sublst == lst[i:i+n]) for i in xrange(len(lst) - n + 1))

def approx_match(label, gold_label, use_includes=False):
    if label == gold_label:
        return True
    # Approximate matching strategy from Zesch and Gurevych (2009).
    # Following their human validation test, we implement the MORPH and
    # INCLUDES matching strategies.
    singularized_label_tokens = [singularize(token) for token in label.split()]
    singularized_gold_label_tokens = [singularize(token) for token in gold_label.split()]
    if use_includes:
        return contains_sublist(singularized_label_tokens, singularized_gold_label_tokens)
    else:
        return singularized_label_tokens == singularized_gold_label_tokens

def evaluate_extractor_on_dataset(extractor, dataset, numExamples):
    reader = READERS[dataset]
    examples = reader()
    tp, fp, fn = 0., 0., 0.
    mistakes = []

    num_correct_r_labels, num_approx_correct_r_labels, total_gold_labels = 0, 0, 0
    
    sys.stdout.write('Evaluating extractor on %s dataset' % (dataset))
    for filename, text, gold_labels in examples[:numExamples]:
        sys.stdout.write('.')
        sys.stdout.flush()

        num_gold_labels = len(gold_labels)
        extracted_labels = extractor(text, num_gold_labels)
        if len(extracted_labels) < num_gold_labels:
            print '\nWARNING: for document %s, extracted %s keyphrases (suggested: %s)' % (filename, len(extracted_labels), num_gold_labels)

        for extracted_label in extracted_labels:
            if extracted_label in gold_labels:
                tp += 1
            else:
                fp += 1

        missed_exact_labels = gold_labels.difference(extracted_labels)
        fn += len(missed_exact_labels)

        # Update R-Precision stats
        total_gold_labels += num_gold_labels
        first_r_labels = set(extracted_labels[:num_gold_labels])
        num_correct_r_labels += len([
            label for label in first_r_labels
            if label in gold_labels
        ])
        approx_labels = set([
            label for label in first_r_labels
            if any([approx_match(label, gold_label) for gold_label in gold_labels])
        ])
        num_approx_correct_r_labels += len(approx_labels)

        # Update mistakes file
        correct_labels = gold_labels.intersection(first_r_labels)
        approx_only_matches = approx_labels.difference(correct_labels)
        missed_labels = [
            gold_label for gold_label in gold_labels
            if not any([approx_match(label, gold_label) for label in first_r_labels])
        ]
        extraneous_labels = first_r_labels.difference(correct_labels).difference(approx_only_matches)
        extra_labels = extracted_labels[num_gold_labels:]

        mistakes.append((filename, gold_labels, correct_labels, approx_only_matches, missed_labels, extraneous_labels, extra_labels))

    print ' Done.'
    r_precision = float(num_correct_r_labels) / total_gold_labels
    r_precision_approx = float(num_approx_correct_r_labels) / total_gold_labels
    return r_precision_approx, r_precision, compute_stats(tp, fp, fn), mistakes

def print_results(results):
    print '\n%-16s%-13s%-13s%-13s%-13s%-13s' % ('Dataset', 'R-P app.', 'R-P ex.', 'Precision', 'Recall', 'F1')
    print '-'*73, '\n'
    for dataset in DATASETS:
        r_precision_approx, r_precision, precision, recall, f1 = results[dataset]
        print '%-16s%-.3f%-8s%-.3f%-8s%-.3f%-8s%-.3f%-8s%-.3f\n' % (
            dataset,
            r_precision_approx, '',
            r_precision, '',
            precision, '',
            recall, '',
            f1
        )
    print '='*73

def output_mistakes(mistakes_list, verbose):
    outputStr = ''
    with open(MISTAKES_FILENAME, 'w') as f:
        for filename, gold_labels, correct_labels, approx_labels, missed_labels, extraneous_labels, extra_labels in mistakes_list:
            outputStr += 'Mistakes for document %s:\n' % (filename)
            outputStr += '-'*50 + '\n'
            outputStr += 'Gold labels -----> %s\n\n' % (', '.join(gold_labels))
            outputStr += 'Correct labels ----> %s\n\n' % (', '.join(correct_labels))
            outputStr += 'Approximately correct labels ----> %s\n\n' % (', '.join(approx_labels))
            outputStr += 'Extraneous labels ----> %s\n\n' % (', '.join(extraneous_labels))
            outputStr += 'Missed labels ----> %s\n\n' % (', '.join(missed_labels))
            outputStr += 'Extra labels ----> %s\n' % (', '.join(extra_labels))
            outputStr += '='*79 + '\n'
        f.write(outputStr.encode('utf-8'))
        if verbose:
            print outputStr

def evaluate_extractor(extractor, numExamples, verbose=False):
    results = {}
    mistakes_list = []
    for dataset in DATASETS:
        r_precision_approx, r_precision, (precision, recall, f1), mistakes = evaluate_extractor_on_dataset(extractor, dataset, numExamples)
        results[dataset] = (r_precision_approx, r_precision, precision, recall, f1)
        mistakes_list += mistakes
    output_mistakes(mistakes_list, verbose)
    print_results(results)

if __name__=='__main__':
    from degreeCentralityModel import DegreeCentralityModel
    from pageRankModel import PageRankModel
    model = DegreeCentralityModel()
    model2 = PageRankModel()
    model2.evaluate()
