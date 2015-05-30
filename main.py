import networkx
from degreeCentralityModel import DegreeCentralityModel
from pageRankModel import PageRankModel
from nltk.corpus import wordnet
import nltk

model = DegreeCentralityModel(windowSize=30, synFilter=[wordnet.NOUN, wordnet.ADJ])
model.evaluate(numExamples=None, verbose=False)
