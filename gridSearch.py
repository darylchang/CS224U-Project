from constants import *
import itertools
from scipy.stats.mstats import gmean
import numpy as np


DATASETS = ['Inspec', 'DUC-2001']
SKIP_DATASETS = ['Handwritten']
MODEL_KEYWORD = 'model'


def myProduct(dicts):
    return (dict(itertools.izip(dicts, x)) for x in itertools.product(*dicts.itervalues()))

def getParamsString(paramCombo):
    lines = []
    for param, value in paramCombo.items():
        lines.append('\t%.30s: %.80s' % (param, value))
    return '\n'.join(lines)

def gridSearch(options, numExamples, verbose=False):
    if MODEL_KEYWORD not in options:
        print 'ERROR: must specify models for grid search under "%s" key.' % (MODEL_KEYWORD)
        return
    paramCombos = myProduct(options)
    bestScore, bestCombo = 0., None

    for paramCombo in paramCombos:
        paramsStr = getParamsString(paramCombo)
        print "Parameters:\n%s" % (paramsStr)
        
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
            paramCombo['ngramAdjacentBoostFn'] = lambda length, count: constant * np.sqrt(length * count) / 50.
            del paramCombo['ngramAdjacentBoostParams']

        constructor = paramCombo[MODEL_KEYWORD]
        del paramCombo[MODEL_KEYWORD]
        model = constructor(**paramCombo)
        paramCombo.update({MODEL_KEYWORD: constructor})

        results = model.evaluate(numExamples, verbose, SKIP_DATASETS)
        score = gmean([results[dataset][0] for dataset in DATASETS if dataset not in SKIP_DATASETS])

        if score > bestScore:
            bestScore, bestCombo = score, paramsStr
        print "Score: {}\n\n\n".format(score)
    
    print 'Best score of %s was achieved by parameters:\n%s' % (bestScore, bestCombo)
