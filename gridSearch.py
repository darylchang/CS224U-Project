import itertools
from scipy.stats.mstats import gmean
import inspect


DATASETS = ['Inspec', 'DUC-2001']
SKIP_DATASETS = ['Handwritten']
MODEL_KEYWORD = 'model'


def myProduct(dicts):
    return (dict(itertools.izip(dicts, x)) for x in itertools.product(*dicts.itervalues()))

def gridSearch(options, numExamples, verbose=False):
    if MODEL_KEYWORD not in options:
        print 'ERROR: must specify models for grid search under "%s" key.' % (MODEL_KEYWORD)
        return
    paramCombos = myProduct(options)
    bestScore, bestCombo = 0., None

    for paramCombo in paramCombos:
        print "Parameters: {}".format(paramCombo)
    
    	# Create length penalty function from params
        power, firstDenom, secondDenom = paramCombo['lengthPenaltyParams']
        paramCombo['lengthPenaltyFn'] = lambda x: x**power/firstDenom if x<4 else x/secondDenom
        del paramCombo['lengthPenaltyParams']

        constructor = paramCombo[MODEL_KEYWORD]
        del paramCombo[MODEL_KEYWORD]
        model = constructor(**paramCombo)
        paramCombo.update({MODEL_KEYWORD: constructor})

        results = model.evaluate(numExamples, verbose, SKIP_DATASETS)
        score = gmean([results[dataset][0] for dataset in DATASETS])

        if score > bestScore:
            bestScore, bestCombo = score, paramCombo
        print "Score: {}\n\n\n".format(score)
    
    print bestScore, bestCombo
