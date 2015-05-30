import cooccurrence
import networkx as nx
from model import BaseModel

class DegreeCentralityModel(BaseModel):

	def extract_keyphrases(self, text, min_num_labels):
		# Tokenize text
		words = self.tokenize(text)

		# Create graph
		cooccurrenceDict = cooccurrence.slidingWindowMatrix(words, self.windowSize, self.synFilter,  self.stripStopWords)
		G = nx.Graph(cooccurrenceDict)

		# Get keywords by node centrality
		node_degrees = nx.degree_centrality(G)
		node_degrees = sorted(node_degrees.items(), key=lambda x:x[1], reverse=True)
		keywords = [keyword for keyword, degree in node_degrees]

		# Combine keywords into keyphrases
		keyphrases, keyphrase = [], []
		for word in words:
			if word in keywords:
				keyphrase.append(word)
			else:
				if keyphrase:
					keyphrases.append(keyphrase)
				keyphrase = []

		result = [' '.join(keyphrase) for keyphrase in keyphrases][:min_num_labels]
		return result

