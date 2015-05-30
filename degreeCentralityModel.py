import cooccurrence
import networkx as nx
from model import BaseModel
import numpy as np

class DegreeCentralityModel(BaseModel):

	def extract_keyphrases(self, text, min_num_labels):
		# Tokenize text
		words = self.tokenize(text) # Note: tuples if using a synFilter

		# Create graph
		cooccurrenceDict = cooccurrence.slidingWindowMatrix(words, self.windowSize, self.synFilter, self.stripStopWords)
		G = nx.Graph(cooccurrenceDict)

		# Get keywords by node centrality
		node_degrees = nx.degree_centrality(G)
		sorted_node_degrees = sorted(node_degrees.items(), key=lambda x:x[1], reverse=True)
		keywords = [keyword for keyword, degree in sorted_node_degrees]
		keyphrases = [(degree, [keyword]) for keyword, degree in sorted_node_degrees]

		# TODO: Look into interleaving nouns/adjs with adverbs and adjectives
		# Combine keywords into keyphrases
		keyphrase = []
		for word in words:
			word = word if not self.synFilter else word[0]
			if word in keywords:
				keyphrase.append(word)
			else:
				if keyphrase:
					score = np.sum([node_degrees[keyword] for keyword in keyphrase])
					keyphrases.append((score, keyphrase))
				keyphrase = []

		keyphrases = sorted(keyphrases, reverse=True)
		result = [' '.join(keyphrase) for score, keyphrase in keyphrases][:min_num_labels]
		return result

