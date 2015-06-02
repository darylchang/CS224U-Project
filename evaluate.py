from baseline import *
from parse import handwritten_data_reader, inspec_data_reader, duc_data_reader
from pattern.en import singularize
import sys
from constants import *

# TODO: Look into using R-precision both token-wise and keyphrase-wise (partial matching, Daryl)

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

def evaluate_extractor_on_dataset(extractor, dataset, numExamples, compute_mistakes):
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
        # Note that the precision is the percentage of gold keyphrases that
        # match an extracted keyphrase. If done the other way around, a system
        # could score artificially high by having slight variants of a single
        # gold keyphrase. But since the gold keyphrases are, by definition,
        # distinct from each other, we don't worry about multiple gold
        # keyphrases matching a single candidate keyphrase.
        total_gold_labels += num_gold_labels
        first_r_labels = set(extracted_labels[:num_gold_labels])
        doc_num_correct_r_labels = len([
            gold_label for gold_label in gold_labels
            if gold_label in first_r_labels
        ])
        doc_num_approx_correct_r_labels = len([
            gold_label for gold_label in gold_labels
            if any([approx_match(label, gold_label) for label in first_r_labels])
        ])
        num_correct_r_labels += doc_num_correct_r_labels
        num_approx_correct_r_labels += doc_num_approx_correct_r_labels

        if compute_mistakes:
            # Update mistakes file
            approx_labels = set([
                label for label in first_r_labels
                if any([approx_match(label, gold_label) for gold_label in gold_labels])
            ])
            correct_labels = gold_labels.intersection(first_r_labels)
            approx_only_matches = approx_labels.difference(correct_labels)
            missed_labels = [
                gold_label for gold_label in gold_labels
                if not any([approx_match(label, gold_label) for label in first_r_labels])
            ]
            extraneous_labels = first_r_labels.difference(correct_labels).difference(approx_only_matches)
            extra_labels = extracted_labels[num_gold_labels:]

            document_performance = float(doc_num_approx_correct_r_labels) / num_gold_labels

            mistakes.append((filename, document_performance, len(text), gold_labels, correct_labels, approx_only_matches, missed_labels, extraneous_labels, extra_labels))

    print ' Done.'
    r_precision = float(num_correct_r_labels) / total_gold_labels
    r_precision_approx = float(num_approx_correct_r_labels) / total_gold_labels
    return r_precision_approx, r_precision, compute_stats(tp, fp, fn), mistakes

def print_results(results, use_datasets):
    print '\n%-16s%-13s%-13s%-13s%-13s%-13s' % ('Dataset', 'R-P app.', 'R-P ex.', 'Precision', 'Recall', 'F1')
    print '-'*73, '\n'
    for dataset in use_datasets:
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

def output_mistakes(mistakes_list, verbose, num_extreme_docs=10):
    outputStr = ''
    with open(MISTAKES_FILENAME, 'w') as f:
        performance = {}
        doc_lengths = {}
        # TODO (anyone): if mistakes output gets slow, this would be faster with a list of strings and a '\n'.join
        for filename, document_performance, doc_length, gold_labels, correct_labels, approx_labels, missed_labels, extraneous_labels, extra_labels in mistakes_list:
            outputStr += 'Mistakes for document %s:\n' % (filename)
            outputStr += '-'*50 + '\n'
            outputStr += 'Gold labels -----> %s\n\n' % (', '.join(gold_labels))
            outputStr += 'Correct labels ----> %s\n\n' % (', '.join(correct_labels))
            outputStr += 'Approximately correct labels ----> %s\n\n' % (', '.join(approx_labels))
            outputStr += 'Extraneous labels ----> %s\n\n' % (', '.join(extraneous_labels))
            outputStr += 'Missed labels ----> %s\n\n' % (', '.join(missed_labels))
            outputStr += 'Extra labels ----> %s\n' % (', '.join(extra_labels))
            outputStr += '='*79 + '\n'
            performance[filename] = document_performance
            doc_lengths[filename] = doc_length
        sorted_performance = sorted(performance.keys(), key=performance.get)
        outputStr += 'Worst documents:\n'
        for filename in sorted_performance[:num_extreme_docs]:
            outputStr += '\t%.3f %s (length: %s)\n' % (performance[filename], filename, doc_lengths[filename])
        outputStr += 'Best documents:\n'
        sorted_performance.reverse()
        for filename in sorted_performance[:num_extreme_docs]:
            outputStr += '\t%.3f %s (length: %s)\n' % (performance[filename], filename, doc_lengths[filename])
        f.write(outputStr.encode('utf-8'))
        if verbose:
            print outputStr

def evaluate_extractor(extractor, numExamples, compute_mistakes=False, verbose=False, use_datasets=DATASETS):
    results = {}
    mistakes_list = []
    for dataset in use_datasets:
        r_precision_approx, r_precision, (precision, recall, f1), mistakes = evaluate_extractor_on_dataset(extractor, dataset, numExamples, compute_mistakes)
        results[dataset] = (r_precision_approx, r_precision, precision, recall, f1)
        mistakes_list += mistakes
    if compute_mistakes:
        output_mistakes(mistakes_list, verbose)
    elif verbose:
        print 'WARNING: cannot print mistakes to terminal if compute_mistakes flag is False.'
    print_results(results, use_datasets)
    return results

if __name__=='__main__':
    from degreeCentralityModel import DegreeCentralityModel
    from pageRankModel import PageRankModel
    model = DegreeCentralityModel()
    model2 = PageRankModel()
    model2.evaluate()
