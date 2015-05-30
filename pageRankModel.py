import cooccurrence
import networkx as nx
from model import BaseModel


class PageRankModel(BaseModel):

	def extract_keyphrases(self, text, min_num_labels):
		# Tokenize text
		words = self.tokenize(text)

		# Create graph
		cooccurrenceDict = cooccurrence.slidingWindowMatrix(words, self.windowSize, self.synFilter, self.stripStopWords)
		G = nx.Graph(cooccurrenceDict)

		# Get keywords by PageRank score
		pagerank_scores = nx.pagerank(G)
		pagerank_scores = sorted(pagerank_scores.items(), key=lambda x:x[1], reverse=True)
		keywords = [keyword for keyword, degree in pagerank_scores][:min_num_labels]
		
		# Combine keywords into keyphrases
		keyphrases, keyphrase = [], []
		for word in words:
			word = word if not self.synFilter else word[0]
			if word in keywords:
				keyphrase.append(word)
			else:
				if keyphrase:
					keyphrases.append(keyphrase)
				keyphrase = []

		result = [' '.join(keyphrase) for keyphrase in keyphrases][:min_num_labels]
		return result
