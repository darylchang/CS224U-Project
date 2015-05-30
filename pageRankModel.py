import cooccurrence
import networkx as nx
from model import BaseModel


class PageRankModel(BaseModel):

	def extract_keywords(self, text, min_num_labels):
		# Tokenize text
		words = self.tokenize(text)

		# Create graph
		cooccurrenceDict = cooccurrence.slidingWindowMatrix(words, 5, self.stripStopWords)
		G = nx.Graph(cooccurrenceDict)

		# Get keywords by PageRank score
		pagerank_scores = nx.pagerank(G)
		pagerank_scores = sorted(pagerank_scores.items(), key=lambda x:x[1], reverse=True)
		keywords = [keyword for keyword, degree in pagerank_scores][:min_num_labels]
		return keywords
