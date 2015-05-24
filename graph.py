import networkx
import numpy as np 
from collections import defaultdict
from parse import tokenize 

####### Create the adjacency matrix first and then a graph based on it #######
#cooccurence by N sized sliding window
#then weight with PMI, or TFIDF 
#RAKE, use delimitors to split the text upon stop word set. Coccurence 
#between word (A,B) is if they occur in the chunks together. 


def slidingWindowMatrix(wordList, N):
	print wordList
	numWords = len(wordList)
	vocab = list(set(wordList))
	vocabSize = len(vocab)
	cooccurenceDict = defaultdict(lambda:0)
	for idx, val in enumerate(wordList):
		beg = idx
		end = beg + N - 1
		if(end>numWords - 1):
			break
		window = list(set(wordList[beg:end+1]))
		window_len = len(window)
		for i1 in range(0, window_len):
			for i2 in range(i1, window_len):
				cooccurenceDict[(window[i1], window[i2])] += 1
	return cooccurenceDict

def RakeMatrix(wordList, delimiters):
	fragments = []
	beg = 0
	for idx, val in enumerate(wordList):
		if wordList[idx] in delimiters:
			frag = wordList[beg:idx]
			fragments.append(frag)
			beg = idx+1
	cooccurenceDict = defaultdict(lambda:0)
	for frag in fragments:
		if len(frag) > 0:
			print frag
			window = list(set(frag))
			window_len = len(window)
			for i1 in range(0, window_len):
				for i2 in range(i1, window_len):
					cooccurenceDict[window[i1], window[i2]] += 1
	return cooccurenceDict, len(cooccurenceDict)

fileName = 'data/Handwritten/yeastsamp.txt'
fileText = open(fileName).read()
# windowWordArray = tokenize(fileText)
# print slidingWindowMatrix(windowWordArray, 5)
rakeWordArray = tokenize(fileText, stripPunct=False)
stopWords = ['a', 'about', 'an', 'are', 'as','at','be','by','for','from','how','in','is','it','of','on','or','that','the','this','to','was','what','when','where','who','will','with','the']
delimiters = set(['.', ','] + stopWords)
print RakeMatrix(rakeWordArray, delimiters)


