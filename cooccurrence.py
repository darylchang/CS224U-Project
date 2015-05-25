import matplotlib.pyplot as plt
import networkx as nx
import numpy as np 
from collections import defaultdict
from parse import tokenize 
from nltk.corpus import stopwords

####### Create the adjacency matrix first and then a graph based on it #######
#cooccurence by N sized sliding window
#then weight with PMI, or TFIDF 
#RAKE, use delimitors to split the text upon stop word set. Coccurence 
#between word (A,B) is if they occur in the chunks together. 
stop = stopwords.words('english')

def slidingWindowMatrix(wordList, N, stripStopWords=True):
	numWords = len(wordList)
	vocab = list(set(wordList))
	vocabSize = len(vocab)
	cooccurrenceDict = defaultdict(lambda: defaultdict(int))
	for idx, val in enumerate(wordList):
		beg = idx
		end = beg + N - 1
		if(end>numWords - 1):
			break
		window = list(set(wordList[beg:end+1]))
		for wordOne in window:
			for wordTwo in window:
				if not stripStopWords:
					cooccurrenceDict[wordOne][wordTwo] += 1
				elif stripStopWords and wordOne not in stop and wordTwo not in stop:
					cooccurrenceDict[wordOne][wordTwo] += 1
	return cooccurrenceDict

def rakeMatrix(wordList, delimiters, stripStopWords=True):
	fragments = []
	beg = 0
	for idx, val in enumerate(wordList):
		if wordList[idx] in delimiters:
			frag = wordList[beg:idx]
			fragments.append(frag)
			beg = idx+1
	cooccurrenceDict = defaultdict(lambda: defaultdict(int))
	for frag in fragments:
		if len(frag) > 0:
			window = list(set(frag))
			for wordOne in window:
				for wordTwo in window:
					if not stripStopWords:
						cooccurrenceDict[wordOne][wordTwo] += 1
					elif stripStopWords and wordOne not in stop and wordTwo not in stop:
						cooccurrenceDict[wordOne][wordTwo] += 1
	return cooccurrenceDict

def main():
	fileName = 'data/Handwritten/yeastsamp.txt'
	fileText = open(fileName).read().decode('utf-8')
	windowWordArray = tokenize(fileText)
	cooccurrenceDict = slidingWindowMatrix(windowWordArray, 5)
	# rakeWordArray = tokenize(fileText, stripPunct=False)
	# stopWords = ['a', 'about', 'an', 'are', 'as','at','be','by','for','from','how','in','is','it','of','on','or','that','the','this','to','was','what','when','where','who','will','with','the']
	# delimiters = set(['.', ','] + stopWords)
	# cooccurrenceDict = RakeMatrix(rakeWordArray, delimiters)
	G = createGraph(cooccurrenceDict)
	labels = nx.draw_networkx_labels(G)
	labels = nx.draw_networkx_edge_labels(G)
	plt.savefig('graph.png')	

if __name__ == '__main__':
	main()

