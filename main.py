import networkx
from degreeCentralityModel import DegreeCentralityModel
from pageRankModel import PageRankModel
from ensembleModel import EnsembleModel
from nltk.corpus import wordnet
import nltk
from gridSearch import gridSearch
import itertools
from scipy.stats.mstats import gmean
from constants import *

model = DegreeCentralityModel(lengthPenaltyFn=lambda x: x**3/60. if x<4 else x/3., useCommunity=False)
results = model.evaluate(numExamples=15, compute_mistakes=True, verbose=False)
score = gmean([results[dataset][0] for dataset in DATASETS if dataset not in SKIP_DATASETS])
print score

# options = dict()
# options['model'] = [DegreeCentralityModel]
# # options['windowSize'] = [22, 24, 26, 28]

# powers = [2,3]
# firstDenoms = [2.,40.,100.,200.]
# secondDenoms = [1.5,2.,3.,4.]

# combos = itertools.product(powers,firstDenoms,secondDenoms)
# options['lengthPenaltyParams'] = combos
# # options['keywordThreshold'] = [2, 3, 4, 5]
# gridSearch(options, numExamples=5, verbose=True)