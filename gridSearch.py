import itertools
from scipy.stats.mstats import gmean

DATASETS = ['Inspec', 'DUC-2001']
SKIP_DATASETS = ['Handwritten']

def myProduct(dicts):
    return (dict(itertools.izip(dicts, x)) for x in itertools.product(*dicts.itervalues()))

def gridSearch(constructor, options, numExamples, verbose=False):
	paramCombos = myProduct(options)
	bestScore, bestCombo = 0., None
	for paramCombo in paramCombos:
		print "Parameters: {}".format(paramCombo)
		model = constructor(**paramCombo)
		results = model.evaluate(numExamples, verbose, SKIP_DATASETS)
		score = gmean([results[dataset][0] for dataset in DATASETS])
		if score > bestScore:
			bestScore, bestCombo = score, paramCombo
		print "Score: {}\n\n\n".format(score)
	print bestScore, bestCombo