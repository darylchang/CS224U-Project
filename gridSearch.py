from constants import *
import itertools
from scipy.stats.mstats import gmean
import numpy as np
from functools import partial
import sys, os
import dill

def myProduct(dicts):
    return (dict(itertools.izip(dicts, x)) for x in itertools.product(*dicts.itervalues()))

def getParamsString(paramCombo):
    lines = []
    for param, value in paramCombo.items():
        lines.append('\t%.30s: %.80s' % (param, value))
    return '\n'.join(lines)

def testCombo(paramCombo, use_datasets, numExamples, compute_mistakes=False, verbose=False, parallelize=True):
    paramsStr = getParamsString(paramCombo)
    if parallelize:
        sys.stdout = open(str(os.getpid()) + ".out", "a")

    # Create length penalty function from params
    if 'lengthPenaltyParams' in paramCombo:
        power, firstDenom, secondDenom = paramCombo['lengthPenaltyParams']
        paramCombo['lengthPenaltyFn'] = lambda x: x**power/firstDenom if x<4 else x/secondDenom
        del paramCombo['lengthPenaltyParams']

    # Create ngram penalty and adjacency boost functions from params
    if 'ngramPenaltyParams' in paramCombo:
        constant = paramCombo['ngramPenaltyParams']
        paramCombo['ngramPenaltyFn'] = lambda length, count: constant * float(length) / np.sqrt(count)
        del paramCombo['ngramPenaltyParams']
    if 'ngramAdjacentBoostParams' in paramCombo:
        constant = paramCombo['ngramAdjacentBoostParams']
        paramCombo['ngramAdjacentBoostFn'] = lambda length, count: constant * np.sqrt(length * count)
        del paramCombo['ngramAdjacentBoostParams']

    constructor = paramCombo[MODEL_KEYWORD]
    del paramCombo[MODEL_KEYWORD]
    model = constructor(**paramCombo)
    paramCombo.update({MODEL_KEYWORD: constructor})

    results = model.evaluate(numExamples=numExamples, compute_mistakes=compute_mistakes, verbose=verbose, use_datasets=use_datasets)
    score = gmean([results[dataset][0] for dataset in use_datasets])

    print "Parameters:\n%s" % (paramsStr)
    print "Score: {}\n\n\n".format(score)
    return score, paramsStr, paramCombo 


def gridSearch(options, use_datasets, numExamples, compute_mistakes=False, verbose=False, parallelize=False):
    if MODEL_KEYWORD not in options:
        print 'ERROR: must specify models for grid search under "%s" key.' % (MODEL_KEYWORD)
        return
    paramCombos = myProduct(options)
    partialTestCombo = partial(testCombo, use_datasets=use_datasets, numExamples=numExamples, compute_mistakes=compute_mistakes, verbose=verbose)
    if parallelize:
        from pathos.multiprocessing import Pool
        p = Pool(5)
        try:
            result = p.map_async(partialTestCombo, paramCombos)
            result = result.get(999999999)
            bestScore, bestParamsStr, bestCombo = max(result, key=lambda x:x[0])
            sys.stdout = open("best.out", "w")
            print 'Best score of %s was achieved by parameters:\n%s' % (bestScore, bestParamsStr)
        except KeyboardInterrupt:
            p.terminate()
            print "You cancelled the program!"
            sys.exit(1)
    else:
        bestScore, bestCombo, bestComboStr = float('-inf'), None, ''
        for paramCombo in paramCombos:
            score, paramsStr, _ = testCombo(paramCombo, use_datasets=use_datasets, numExamples=numExamples, compute_mistakes=compute_mistakes, verbose=verbose, parallelize=False)
            if score > bestScore:
                bestScore, bestCombo, bestComboStr = score, paramCombo, paramsStr
        print 'Best score of %s was achieved by parameters:\n%s' % (bestScore, bestComboStr)