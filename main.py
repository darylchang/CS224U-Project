import networkx
from degreeCentralityModel import DegreeCentralityModel
from pageRankModel import PageRankModel
from nltk.corpus import wordnet
import nltk
from gridSearch import gridSearch
import itertools

model = DegreeCentralityModel(lengthPenaltyFn=lambda x: x**3/60. if x<4 else x/3.)
model.evaluate(numExamples=5, compute_mistakes=True, verbose=False)

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