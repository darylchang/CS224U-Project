import networkx
from degreeCentralityModel import DegreeCentralityModel
from pageRankModel import PageRankModel
from nltk.corpus import wordnet
import nltk

model = DegreeCentralityModel(windowSize=5, synFilter=[wordnet.NOUN, wordnet.ADJ])
model.evaluate(numExamples=100, verbose=True)