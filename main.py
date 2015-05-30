import networkx
from degreeCentralityModel import DegreeCentralityModel
from nltk.corpus import wordnet
import nltk

model = DegreeCentralityModel(windowSize=5)
model.evaluate(numExamples=5)