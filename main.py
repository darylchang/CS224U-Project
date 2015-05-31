import networkx
from degreeCentralityModel import DegreeCentralityModel
from pageRankModel import PageRankModel
from nltk.corpus import wordnet
import nltk
from gridSearch import gridSearch

# model = DegreeCentralityModel(windowSize=10, synFilter=[wordnet.NOUN, wordnet.ADJ], 
							  # keywordThreshold=5)
# model.evaluate(numExamples=10, verbose=True)

options = dict()
options['windowSize']=range(24, 33, 2)
options['keywordThreshold']=[3, 4, 5]
gridSearch(PageRankModel, options, numExamples=3, verbose=True)