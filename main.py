import networkx
from degreeCentralityModel import DegreeCentralityModel
from pageRankModel import PageRankModel
import ngrams
from nltk.corpus import wordnet
import nltk
import numpy as np
from gridSearch import gridSearch
import itertools


bigrams = ngrams.read_bigrams()
trigrams = ngrams.read_trigrams()

lengthPenaltyFn = lambda x: x**3/60. if x<4 else x/3.
ngramPenaltyFn = lambda length, count: 0.4 * float(length) / np.sqrt(count)
ngramAdjacentBoostFn = lambda length, count: np.sqrt(length * count) / 50.

model = DegreeCentralityModel(lengthPenaltyFn=lengthPenaltyFn, useNgrams=[bigrams, trigrams], ngramPenaltyFn=ngramPenaltyFn, ngramAdjacentBoostFn=ngramAdjacentBoostFn, keywordThreshold=10)
model.evaluate(numExamples=10, compute_mistakes=True, verbose=False)

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