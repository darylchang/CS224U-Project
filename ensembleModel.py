import cooccurrence
import networkx as nx
from model import BaseModel
import numpy as np


class EnsembleModel(BaseModel):

	def extract_keyphrases(self, text, min_num_labels):
		# Tokenize text
		words = self.tokenize(text) # Note: tuples if using a synFilter
		G = self.create_graph(words)

		# Get keywords by node centrality and pagerank score
		degree_scores = nx.degree_centrality(G)
		pagerank_scores = nx.pagerank(G)
		hybrid_scores = {k: degree_scores[k] + 10 * pagerank_scores[k] for k in degree_scores}

		return self.combine_to_keyphrases(text, words, hybrid_scores, min_num_labels)