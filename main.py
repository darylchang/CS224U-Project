import networkx
from degreeCentralityModel import DegreeCentralityModel
from pageRankModel import PageRankModel
from nltk.corpus import wordnet
import nltk
from gridSearch import gridSearch

# model = DegreeCentralityModel(windowSize=10, synFilter=[wordnet.NOUN, wordnet.ADJ], 
# 							  keywordThreshold=5)
# model.evaluate(numExamples=10, verbose=False)

options = dict()
options['windowSize']=[5,10,15]
options['keywordThreshold']=[4,5,6]
options 
gridSearch(PageRankModel, options, numExamples=3, verbose=True)