import cooccurrence
import networkx as nx
from model import BaseModel


class DegreeCentralityModel(BaseModel):

	def extract_keywords(self, text, min_num_labels):
		# Tokenize text
		words = self.tokenize(text)

		# Create graph
		cooccurrenceDict = cooccurrence.slidingWindowMatrix(words, 50, self.stripStopWords)
		G = nx.Graph(cooccurrenceDict)

		# Get keywords by node centrality
		node_degrees = nx.degree_centrality(G)
		node_degrees = sorted(node_degrees.items(), key=lambda x:x[1], reverse=True)
		keywords = [keyword for keyword, degree in node_degrees][:min_num_labels]
		return keywords