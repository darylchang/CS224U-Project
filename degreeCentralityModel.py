import cooccurrence
import networkx as nx
from model import BaseModel
import numpy as np
import community
from collections import defaultdict

class DegreeCentralityModel(BaseModel):

    def extract_keyphrases(self, text, min_num_labels):
        # Tokenize text
        taggedWords = self.preprocess(text) # Note: tuples if using a synFilter
        cooccurrence_dict, G = self.create_graph(taggedWords)

        # Get keywords by node centrality
        scores = nx.degree_centrality(G)

        # Community detection using Louvain algorithm
        # if self.useCommunity:
        #     partition = community.best_partition(G)
        #     clusters = defaultdict(list)
        #     for node, cluster_id in partition.items():
        #         clusters[cluster_id].append(node)
        #     scores = [{node: scores[node] for node in clusters[cluster_id]} for cluster_id in clusters]

        return self.combine_to_keyphrases(text, taggedWords, scores, min_num_labels)
