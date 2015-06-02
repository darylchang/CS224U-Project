import networkx
from degreeCentralityModel import DegreeCentralityModel
from pageRankModel import PageRankModel
import ngrams
from ensembleModel import EnsembleModel
from nltk.corpus import wordnet
import nltk
import numpy as np
from gridSearch import gridSearch
import itertools
from scipy.stats.mstats import gmean
from constants import *


bigrams = ngrams.read_bigrams()
trigrams = ngrams.read_trigrams()

# model = DegreeCentralityModel(lengthPenaltyFn=lambda x: x**3/60. if x<4 else x/3., useCommunity=True)
# results = model.evaluate(numExamples=5, compute_mistakes=True, verbose=False)


lengthPenaltyFn = lambda x: x**3/60. if x<4 else x/3.
ngramPenaltyFn = lambda length, count: 0.3 * float(length) / np.sqrt(count)
ngramAdjacentBoostFn = lambda length, count: np.sqrt(length * count) / 50.

model = DegreeCentralityModel(lengthPenaltyFn=lengthPenaltyFn, useNgrams=[bigrams, trigrams], ngramPenaltyFn=ngramPenaltyFn, ngramAdjacentBoostFn=ngramAdjacentBoostFn, keywordThreshold=10)
model.evaluate(numExamples=50, compute_mistakes=True, verbose=False)


# options = dict()
# options['model'] = [DegreeCentralityModel]
# # options['windowSize'] = [22, 24, 26, 28]

# powers = [3]
# firstDenoms = [40.]
# secondDenoms = [2.]

# combos = itertools.product(powers,firstDenoms,secondDenoms)
# options['lengthPenaltyParams'] = combos
# # options['keywordThreshold'] = [2, 3, 4, 5]

# options['useNgrams'] = [[bigrams, trigrams]]
# options['ngramPenaltyParams'] = [0.25,0.5]
# options['ngramAdjacentBoostParams'] = [1/50.]

# use_datasets = ['Inspec', 'DUC-2001']
# gridSearch(options, use_datasets=use_datasets, numExamples=5, verbose=True)
