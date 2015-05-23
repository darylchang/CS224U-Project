import networkx
import numpy as np 
from collections import defaultdict

####### Create the adjacency matrix first and then a graph based on it #######
#cooccurence by N sized sliding window
#then weight with PMI, or TFIDF 
#RAKE, use delimitors to split the text upon stop word set. Coccurence 
#between word (A,B) is if they occur in the chunks together. 


def textToWordArray(fileName):
	wordArray = []
	with open(fileName) as f:
		for line in f:
			lineparse = line.lower().split()
			for word in lineparse:
				wordArray.append(word.strip('.').strip(','))
	return wordArray

def slidingWindowMatrix(wordList, N):
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



fileName = 'yeastsamp.txt'
wordArray = textToWordArray(fileName)
#print(wordArray)
print slidingWindowMatrix(wordArray, 5)