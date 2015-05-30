import networkx
from degreeCentralityModel import DegreeCentralityModel
from nltk.corpus import wordnet
import nltk

model = DegreeCentralityModel(synFilter=[wordnet.NOUN, wordnet.ADJ])
model.evaluate(numExamples=5)