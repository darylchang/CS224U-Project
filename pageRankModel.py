import cooccurrence
import networkx as nx
from model import BaseModel
import numpy as np


class PageRankModel(BaseModel):

    def extract_keyphrases(self, text, min_num_labels):
        # Tokenize text and create graph
        words = self.tokenize(text) # Note: tuples if using a synFilter
        G = self.create_graph(words)

        # Get keywords by node centrality
        scores = nx.pagerank(G, max_iter=500)
        if 'linear diophantine equations' in text:
            print
            for keyword in sorted(scores.keys(), key=scores.get):
                print '%s: %s' % (keyword, scores[keyword])
        return self.combine_to_keyphrases(text, words, [scores], min_num_labels)
