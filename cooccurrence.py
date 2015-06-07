import matplotlib.pyplot as plt
import networkx as nx
import numpy as np 
from collections import defaultdict
from nltk.corpus import stopwords

####### Create the adjacency matrix first and then a graph based on it #######
# cooccurence by N sized sliding window
# then weight with PMI, or TFIDF 
# RAKE, use delimitors to split the text upon stop word set. Coccurence 
# between word (A,B) is if they occur in the chunks together. 
STOP = stopwords.words('english')


def findNgrams(inputList, N=None):
  if N:
    N = min(N, len(inputList))
    return zip(*[inputList[i:] for i in range(N)])
  else:
    return [phrase for n in range(1,len(inputList)+1) for phrase in findNgrams(inputList,n)]

def getSynFilteredNgram(synFilter, nGram):
    if not synFilter:
        return nGram
    words = []
    for word, tag in nGram:
        if tag in synFilter:
            words.append(word)
    return words

# TODO (all): Consider scoring using multiple window sizes, or else really
#             limiting window size? For large N on the Inspec data set, there
#             are very few ngrams created.
def slidingWindowMatrix(words, N, synFilter, stripStopWords=True, TODOdelete=False):  
    cooccurrenceDict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    nGrams = findNgrams(words, N)
    for nGram in nGrams:
        if TODOdelete:
            print '\t%s' % (str(nGram))
        nGram = getSynFilteredNgram(synFilter, nGram)
        if TODOdelete:
            print '\t\t%s' % (str(nGram))
        for wordOne in nGram:
            for wordTwo in nGram:
                if wordOne != wordTwo:
                    if TODOdelete:
                        print '%s and %s co-occur' % (wordOne, wordTwo)
                    if not stripStopWords:
                        cooccurrenceDict[wordOne][wordTwo]['weight'] += 1.
                    elif stripStopWords and wordOne not in STOP and wordTwo not in STOP:
                        cooccurrenceDict[wordOne][wordTwo]['weight'] += 1.
    return cooccurrenceDict

def rakeMatrix(words, delimiters, synFilter, stripStopWords=True):
    # Create fragments
    fragments, frag = []
    for word in words:
        if word in delimiters:
            fragments.append(frag)
            frag = []
        else:
            frag.append(word)
    if frag:
        fragments.append(frag)

    # TODO: Explore using set() for word occcurrences in window definition
    cooccurrenceDict = defaultdict(lambda: defaultdict(int))
    for frag in fragments: 
        frag = getSynFilteredNgram(synFilter, frag)
        for wordOne in frag:
            for wordTwo in frag:
                if not stripStopWords:
                    cooccurrenceDict[wordOne][wordTwo] += 1
                elif stripStopWords and wordOne not in STOP and wordTwo not in STOP:
                    cooccurrenceDict[wordOne][wordTwo] += 1
    return cooccurrenceDict

# def main():
#   fileName = 'data/Handwritten/yeastsamp.txt'
#   fileText = open(fileName).read().decode('utf-8')
#   windowWordArray = tokenize(fileText)
#   cooccurrenceDict = slidingWindowMatrix(windowWordArray, 5)
#   # rakeWordArray = tokenize(fileText, stripPunct=False)
#   # stopWords = ['a', 'about', 'an', 'are', 'as','at','be','by','for','from','how','in','is','it','of','on','or','that','the','this','to','was','what','when','where','who','will','with','the']
#   # delimiters = set(['.', ','] + stopWords)
#   # cooccurrenceDict = RakeMatrix(rakeWordArray, delimiters)
#   G = createGraph(cooccurrenceDict)
#   labels = nx.draw_networkx_labels(G)
#   labels = nx.draw_networkx_edge_labels(G)
#   plt.savefig('graph.png')    

if __name__ == '__main__':
    main()
