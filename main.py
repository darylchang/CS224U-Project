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

model = PageRankModel(lengthPenaltyFn=lambda x: x**3/200. if x<4 else x/3., windowSize=6, lemmatize=False)
model.evaluate(numExamples=5, compute_mistakes=True, verbose=True)

# lengthPenaltyFn = lambda x: x**3/60. if x<4 else x/3.
# ngramPenaltyFn = lambda length, count: 0.3 * float(length) / np.sqrt(count)
# ngramAdjacentBoostFn = lambda length, count: np.sqrt(length * count) / 50.

# model = DegreeCentralityModel(lengthPenaltyFn=lengthPenaltyFn, useNgrams=[bigrams, trigrams], ngramPenaltyFn=ngramPenaltyFn, ngramAdjacentBoostFn=ngramAdjacentBoostFn, keywordThreshold=10)
# model.evaluate(numExamples=50, compute_mistakes=True, verbose=False)


# options = dict()
# options['model'] = [PageRankModel]
# options['useNgrams'] = [[]]


# options['windowSize'] = [4]
# options['keywordThreshold'] = [6]
# options['ngramPenaltyParams'] = [0.05]
# options['ngramAdjacentBoostParams'] = [1/75.]
# powers = [3]
# firstDenoms = [15000.]
# secondDenoms = [110.]

# # Combine length penalty parameters
# combos = itertools.product(powers,firstDenoms,secondDenoms)
# options['lengthPenaltyParams'] = combos
# use_datasets = [INSPEC_DATASET]
# gridSearch(options, use_datasets=use_datasets, numExamples=2, compute_mistakes=True, verbose=True)
