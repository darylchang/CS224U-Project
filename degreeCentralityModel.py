import cooccurrence
import networkx as nx
from model import BaseModel
import numpy as np
import community
from collections import defaultdict

class DegreeCentralityModel(BaseModel):

	def extract_keyphrases(self, text, min_num_labels):
		# Tokenize text
		words = self.tokenize(text) # Note: tuples if using a synFilter
		G = self.create_graph(words)

		# Get keywords by node centrality
		scores = nx.degree_centrality(G)

		# Community detection using Louvain algorithm
		if self.useCommunity:
			partition = community.best_partition(G)
			clusters = defaultdict(list)
			for node, cluster_id in partition.items():
				clusters[cluster_id].append(node)
			scores = [{node: scores[node] for node in clusters[cluster_id]} for cluster_id in clusters]
		else:
			scores = [scores]

		return self.combine_to_keyphrases(text, words, scores, min_num_labels)