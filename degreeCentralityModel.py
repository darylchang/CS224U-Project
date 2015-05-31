import cooccurrence
import networkx as nx
from model import BaseModel
import numpy as np


class DegreeCentralityModel(BaseModel):

	def extract_keyphrases(self, text, min_num_labels):
		# Tokenize text
		words = self.tokenize(text) # Note: tuples if using a synFilter
		G = self.create_graph(words)

		# Get keywords by node centrality
		scores = nx.degree_centrality(G)
		return self.combine_to_keyphrases(text, words, scores, min_num_labels)