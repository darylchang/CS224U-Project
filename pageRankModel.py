import cooccurrence
import networkx as nx
from model import BaseModel
from collections import Counter
import numpy as np


class PageRankModel(BaseModel):

    def custom_pagerank(self, cooccurrenceDict, damping_factor=0.85, max_iter=500, convergence=0.00001, print_progress=False):
        keywords = set()
        for w1 in cooccurrenceDict:
            keywords.add(w1)
            for w2 in cooccurrenceDict[w1]:
                keywords.add(w2)
        scores = Counter({keyword: 1. for keyword in keywords})

        for iteration in xrange(max_iter):
            changes = []
            new_scores = {}
            for w1 in keywords:
                new_score = 0.
                for w2 in cooccurrenceDict[w1]:
                    new_score += scores[w2] / len(cooccurrenceDict[w2])
                new_score = 1. - damping_factor + damping_factor * new_score
                new_scores[w1] = new_score
                changes.append(abs(scores[w1] - new_score))

            standard_error = np.std(changes) / np.sqrt(len(changes))
            scores = new_scores
            if standard_error < convergence:
                return scores
            elif print_progress:
                print 'Iteration %s: standard error is %s' % (iteration, standard_error)
        
        raise Exception('Pagerank did not converge in %s iterations' % (max_iter))

    def extract_keyphrases(self, text, min_num_labels):
        # Tokenize text and create graph
        words = self.preprocess(text) # Note: tuples if using a synFilter
        cooccurrenceDict, G = self.create_graph(words)

        # Get keywords by node centrality
        scores = nx.pagerank(G, max_iter=500)
        return self.combine_to_keyphrases(words, scores, min_num_labels)