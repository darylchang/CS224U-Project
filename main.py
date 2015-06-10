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

# model = DegreeCentralityModel(lengthPenaltyFn=lambda x: x**3/180. if x<4 else x/3., windowSize=6, lemmatize=False)
# model.evaluate(numExamples=50, compute_mistakes=True, verbose=True)

# lengthPenaltyFn = lambda x: x**3/60. if x<4 else x/3.
# ngramPenaltyFn = lambda length, count: 0.3 * float(length) / np.sqrt(count)
# ngramAdjacentBoostFn = lambda length, count: np.sqrt(length * count) / 50.

# model = DegreeCentralityModel(lengthPenaltyFn=lengthPenaltyFn, useNgrams=[bigrams, trigrams], ngramPenaltyFn=ngramPenaltyFn, ngramAdjacentBoostFn=ngramAdjacentBoostFn, keywordThreshold=10)
# model.evaluate(numExamples=50, compute_mistakes=True, verbose=False)


options = dict()
options['model'] = [DegreeCentralityModel]
options['useNgrams'] = [[bigrams, trigrams]]

options['windowSize'] = [3, 6]
options['keywordThreshold'] = [3, 5]

options['ngramPenaltyParams'] = [0.05, 0.2]
options['ngramAdjacentBoostParams'] = [1/60.]

powers = [3]
firstDenoms = [100, 250, 1000]
secondDenoms = [5., 10.]

combos = itertools.product(powers,firstDenoms,secondDenoms)
options['lengthPenaltyParams'] = combos
use_datasets = [INSPEC_DATASET]
gridSearch(options, use_datasets=use_datasets, numExamples=None, verbose=False)