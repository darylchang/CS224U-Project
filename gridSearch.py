import itertools

def myProduct(dicts):
    return (dict(itertools.izip(dicts, x)) for x in itertools.product(*dicts.itervalues()))

def gridSearch(constructor, numExamples, options):
	paramCombos = myProduct(options)
	for paramCombo in paramCombos:
		model = constructor(**paramCombo)
		model.evaluate(numExamples)