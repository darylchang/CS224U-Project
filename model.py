import networkx as nx
import cooccurrence
import evaluate
from parse import tokenize

class DegreeCentralityModel:

	def __init__(self):
		self.stripStopWords=True
		self.lemmatize=False

	def extract_keywords(self, text):
		# Tokenize text
		tokenize(text, lemmatize=self.lemmatize)

		# Create graph
		cooccurrenceDict = cooccurrence.slidingWindowMatrix(words, 50, self.stripStopWords)
		G = nx.Graph(cooccurrenceDict)

		# Get keywords by node centrality
		node_degrees = nx.degree_centrality(G)
		node_degrees = sorted(node_degrees.items(), key=lambda x:x[1], reverse=True)
		keywords = [keyword for keyword, degree in node_degrees][:5]
		print keywords
		return keywords

	def evaluate(self):
		evaluate.evaluate_extractor(self, self.lemmatize)


class PageRankModel:

	def __init__(self):
		self.stripStopWords=True
		self.lemmatize=False

	def extract_keywords(self, text):
		# Tokenize text
		tokenize(text, lemmatize=self.lemmatize)

		# Create graph
		cooccurrenceDict = cooccurrence.slidingWindowMatrix(words, 50, self.stripStopWords)
		G = nx.Graph(cooccurrenceDict)

		node_degrees = sorted(node_degrees.items(), key=lambda x:x[1], reverse=True)
		keywords = [keyword for keyword, degree in node_degrees][:5]
		return keywords

	def evaluate(self):
		evaluate.evaluate_extractor(self, self.lemmatize)

if __name__=="__main__":
	model = DegreeCentralityModel()
	model.evaluate()