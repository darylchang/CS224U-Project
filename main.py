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
from multiprocessing import Pool
import dill

bigrams = ngrams.read_bigrams()
trigrams = ngrams.read_trigrams()

lengthPenaltyFn = lambda x: x**3/100. if x<4 else x/10.
ngramPenaltyFn = lambda length, count: 0.05 * float(length) / np.sqrt(count)
ngramAdjacentBoostFn = lambda length, count: np.sqrt(length * count) / 0.016667

model = DegreeCentralityModel(lengthPenaltyFn=lengthPenaltyFn, useNgrams=[bigrams, trigrams], ngramPenaltyFn=ngramPenaltyFn, ngramAdjacentBoostFn=ngramAdjacentBoostFn, keywordThreshold=5)
model.evaluate(numExamples=5, compute_mistakes=True, verbose=False)

# options = dict()
# options['model'] = [DegreeCentralityModel]
# options['useNgrams'] = [[bigrams, trigrams]]

# options['windowSize'] = [5, 6, 7]
# options['keywordThreshold'] = [2, 3]

# options['ngramPenaltyParams'] = [0.05, 0.1]
# options['ngramAdjacentBoostParams'] = [1/60.]

# powers = [3]
# firstDenoms = [100]
# secondDenoms = [5., 10.]

# combos = itertools.product(powers,firstDenoms,secondDenoms)
# options['lengthPenaltyParams'] = combos
# use_datasets = [DUC_DATASET]
# gridSearch(options, use_datasets=use_datasets, numExamples=None, verbose=True, parallelize=True)
