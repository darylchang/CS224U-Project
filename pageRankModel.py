from model import BaseModel

class PageRankModel(BaseModel):

	def __init__(self):
		self.stripStopWords=True
		self.lemmatize=False

	def extract_keywords(self, text):
		# Tokenize text
		tokenize(text, lemmatize=self.lemmatize)

		# Create graph
		cooccurrenceDict = cooccurrence.slidingWindowMatrix(words, 50, self.stripStopWords)
		G = nx.Graph(cooccurrenceDict)

		node_degrees = sorted(node_degrees.items(), key=lambda x:x[1], reverse=True)
		keywords = [keyword for keyword, degree in node_degrees][:5]
		return keywords

	def evaluate(self):
		evaluate.evaluate_extractor(self)