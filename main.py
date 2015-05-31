import networkx
from degreeCentralityModel import DegreeCentralityModel
from pageRankModel import PageRankModel
from nltk.corpus import wordnet
import nltk
from gridSearch import gridSearch

# model = DegreeCentralityModel(windowSize=10, synFilter=[wordnet.NOUN, wordnet.ADJ], 
# 							  keywordThreshold=5)
# model.evaluate(numExamples=3, verbose=True)

options = dict()
options['model'] = [DegreeCentralityModel]
options['windowSize'] = [22, 24, 26, 28]
options['lengthPenaltyFn'] = [lambda x: x / 50.0]
options['keywordThreshold'] = [2, 3, 4, 5]
gridSearch(options, numExamples=10, verbose=True)