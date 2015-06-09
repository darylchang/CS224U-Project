from constants import *
import cooccurrence
import evaluate
import ngrams
import networkx as nx
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
import numpy as np
from parse import *
from matplotlib import pyplot as plt
from subprocess import *


class BaseModel:

    def __init__(self,
        stripPunct=True,
        stemRule=None,
        lemmatize=False,
        stripStopWords=True,
        synFilter=[wordnet.NOUN, wordnet.ADJ], 
        windowSize=24,
        keywordThreshold=5,
        numExtraLabels=5,
        lengthPenaltyFn=None,
        useCommunity=False,
        useNgrams=[],
        ngramPenaltyFn=None,
        ngramAdjacentBoostFn=None,
    ):
        self.stripPunct = stripPunct
        self.stemRule = stemRule
        self.lemmatize = lemmatize
        self.stripStopWords = stripStopWords
        self.synFilter = synFilter
        self.windowSize = windowSize
        self.keywordThreshold = keywordThreshold
        self.numExtraLabels = numExtraLabels
        self.lengthPenaltyFn = lengthPenaltyFn
        self.useCommunity = useCommunity
        self.useNgrams = useNgrams
        self.ngramPenaltyFn = ngramPenaltyFn
        self.ngramAdjacentBoostFn = ngramAdjacentBoostFn

    def tokenize(self, text):
        # Strip punctuation if unneeded for co-occurrence counts
        if self.stripPunct:
            pattern = r'''(?x)           # set flag to allow verbose regexps
                      ([A-Z]\.)+         # abbreviations, e.g. U.S.A.
                      | \-?[^\W_]+([-'][^\W_]+)*    # words w/ optional internal hyphens/apostrophe
                      | \$?\d+(\.\d+)?%? # numbers, incl. currency and percentages
                      | [+/\-@&*]        # special characters with meanings
            '''
            tokenizer = RegexpTokenizer(pattern)
            tokens = tokenizer.tokenize(text)
        else:
            tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

        # Normalize words
        words = [t.lower() for t in tokens]
        
        # Stemming. Faster than lemmatization but imprecise
        if self.stemRule:
            stemmer = nltk.PorterStemmer() if self.stemRule=='Porter' else nltk.LancasterStemmer()
            words = [stemmer.stem(w) for w in words]

        # POS tagging
        if self.lemmatize or self.synFilter:
            #taggedWords = self.tag(words)
            taggedWords = [(word, self.wordnetPosCode(tag)) for word, tag in nltk.pos_tag(words)]

        # Lemmatize
        if self.lemmatize:
        	lemmatizer = nltk.WordNetLemmatizer()
        	words = [lemmatizer.lemmatize(word, tag) for word, tag in taggedWords]

        return taggedWords if self.synFilter else words

    def tag(self, words):
        words = ' '.join(words).encode('utf-8').strip() # Avoid ascii codec error
        command = "apache-opennlp/bin/opennlp POSTagger apache-opennlp/models/en-pos-perceptron.bin"
        p = Popen(command, shell=True, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        output = p.communicate(input=words)[0].split('\n')[1]
        result = []
        for token in output.split():
            word, tag = token.split('_')
            result.append((word, self.wordnetPosCode(tag)))
        return result

    # Maps from NLTK POS tags to WordNet POS tags
    def wordnetPosCode(self, tag):
        if tag.startswith('NN'):
            return wordnet.NOUN
        elif tag.startswith('VB'):
            return wordnet.VERB
        elif tag.startswith('JJ'):
            return wordnet.ADJ
        elif tag.startswith('RB'):
            return wordnet.ADV
        else:
            return ''

    def create_graph(self, words, TODOdelete=False):
        cooccurrenceDict = cooccurrence.slidingWindowMatrix(words, self.windowSize, self.synFilter, self.stripStopWords, TODOdelete=TODOdelete)
        return cooccurrenceDict, nx.Graph(cooccurrenceDict)

    def addCommonNgramsAndScores(self, keywords, wordScores, keyphraseScores):
        for ngramsCounter in self.useNgrams:
            for ngramWords, ngramCount in ngrams.get_matching_ngrams(ngramsCounter, keywords):
                ngramScore = sum([wordScores[word] for word in ngramWords])

                if self.ngramPenaltyFn and ngramWords not in keyphraseScores:
                    # print 'Subtracting %s ngram penalty from score of %s for ngram: %s' % (self.ngramPenaltyFn(len(ngramWords), ngramCount), ngramScore, ' '.join(ngramWords))
                    ngramScore -= self.ngramPenaltyFn(len(ngramWords), ngramCount)
                if self.ngramAdjacentBoostFn and ngramWords in keyphraseScores:
                    # print 'Adding %s adjacent ngram boost to score of %s for ngram: %s' % (self.ngramAdjacentBoostFn(len(ngramWords), ngramCount), ngramScore, ' '.join(ngramWords))
                    ngramScore += self.ngramAdjacentBoostFn(len(ngramWords), ngramCount)
                if self.lengthPenaltyFn:
                    ngramScore -= self.lengthPenaltyFn(len(ngramWords))

                if ngramWords not in keyphraseScores or ngramScore > keyphraseScores[ngramWords]:
                    keyphraseScores[ngramWords] = ngramScore

    # TODO (Daryl): Look into interleaving nouns/adjs with adverbs and other POS
    def combine_to_keyphrases(self, text, words, scores_list, min_num_labels):
        combinedKeyphraseScores = {}
        for scores in scores_list:
            sortedScores = sorted(scores.items(), key=lambda x:x[1], reverse=True)[:int(self.keywordThreshold*min_num_labels)]
            keywords = set([keyword for keyword, score in sortedScores])
            keyphraseScores = {(keyword,): scores[keyword] for keyword in keywords}
            if self.lengthPenaltyFn:
                for singleKeyword in keyphraseScores:
                    keyphraseScores[singleKeyword] -= self.lengthPenaltyFn(1)

            # Combine keywords into keyphrases
            keyphrase = ()
            for word in words:
                word = word if not self.synFilter else word[0]
                if word in keywords:
                    keyphrase += (word,)
                else:
                    candidateKeyphrases = cooccurrence.findNgrams(keyphrase)
                    for candidateKeyphrase in candidateKeyphrases:
                        if candidateKeyphrase and ' '.join(candidateKeyphrase) in text:
                            score = sum([scores[keyword] for keyword in candidateKeyphrase])
                            if self.lengthPenaltyFn:
                                # print 'Length penalty: %s being reduced from %s for length of %s' % (self.lengthPenaltyFn(len(keyphrase)), score, len(keyphrase))
                                score -= self.lengthPenaltyFn(len(candidateKeyphrase))
                            keyphraseScores[candidateKeyphrase] = score
                    keyphrase = ()

            if self.useNgrams:
                self.addCommonNgramsAndScores(keywords, scores, keyphraseScores)

            # Add this set of keyphrase scores to the combined score dict,
            # resolving conflicts by taking the higher score.
            # TODO (all): Experiment with combinations other than max?
            for keyphrase in keyphraseScores:
                combinedKeyphraseScores[keyphrase] = max(
                    combinedKeyphraseScores[keyphrase],
                    keyphraseScores[keyphrase],
                ) if keyphrase in combinedKeyphraseScores else keyphraseScores[keyphrase]

        keyphrases = sorted(combinedKeyphraseScores.keys(), key=combinedKeyphraseScores.get, reverse=True)
        result = [' '.join(keyphrase) for keyphrase in keyphrases][:min_num_labels+self.numExtraLabels]
        return result


    def extract_keyphrases(self, text, min_num_labels):
        raise NotImplementedError

    def evaluate(self, numExamples=None, compute_mistakes=False, verbose=False, use_datasets=DATASETS):
        return evaluate.evaluate_extractor(self.extract_keyphrases, numExamples, compute_mistakes, verbose, use_datasets)

    def draw_graph(self, docName):
        examples = inspec_data_reader()
        docText = [text for filename, text, gold_labels in examples if filename == docName][0]
        words = self.tokenize(docText) # Note: tuples if using a synFilter
        cooccurrence_dict, G = self.create_graph(words)
        G = G.to_undirected()
        labels = nx.draw_graphviz(G, with_labels=True, edge_color='0.5', width=0.5, font_color='blue', node_color='0.5', node_size=800)
        plt.show() 
