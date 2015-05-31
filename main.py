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
options['windowSize']=[4,5,6,7]
options['keywordThreshold']=[2,3,4]
gridSearch(PageRankModel, options, numExamples=10, verbose=True)