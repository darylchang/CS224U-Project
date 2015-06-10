from constants import *
import cooccurrence
import evaluate
import ngrams
import networkx as nx
import nltk
from nltk.corpus import wordnet
import numpy as np
from parse import *
from matplotlib import pyplot as plt
from subprocess import *
from collections import defaultdict
import re


class BaseModel:

    def __init__(self,
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
        self.lemmasToWords = defaultdict(list)
        self.wordsToLemmas = defaultdict(str)

    def preprocess(self, text):
        # Tokenize input
        self.taggedWords = [tuple(taggedWord.split('_')) for taggedWord in text.split() 
                            if not re.findall(r'[\W]', taggedWord[0])
                            and len(taggedWord.split('_'))==2]
        self.text = ' '.join([token.split('_')[0] for token in text.split()])

        # Lemmatize
        if self.lemmatize:
            lemmatizer = nltk.WordNetLemmatizer()
            self.taggedLemmas = []
            self.lemmasToWords = defaultdict(list)
            self.wordsToLemmas = defaultdict(list)

            for word, tag in self.taggedWords:
                lemma = lemmatizer.lemmatize(word, tag) if tag else word
                self.lemmasToWords[lemma].append(word)
                self.wordsToLemmas[word].append(lemma)
                self.taggedLemmas.append((lemma,tag))
            return self.taggedLemmas
        else:
            return self.taggedWords

    def tag(self, words):
        words = ' '.join(words).encode('utf-8').strip() # Avoid ascii codec error
        command = "apache-opennlp/bin/opennlp POSTagger apache-opennlp/models/en-pos-perceptron.bin"
        p = Popen(command, shell=True, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        output = p.communicate(input=words)[0].split('\n')[1]
        result = []
        for token in output.split():
            if token.endswith('_``'):
                # print '\nRetrying POS tagging for %s' % (token)
                p = Popen(command, shell=True, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
                token = p.communicate(input=token.split('_')[0])[0].split('\n')[1]
                # print '\tGot new token: %s' % (token)
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

    def create_graph(self, words):
        cooccurrenceDict = cooccurrence.slidingWindowMatrix(words, self.windowSize, self.synFilter, self.stripStopWords)
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

    # Gets best score for a keyword (mainly for use with lemmatization)
    def get_best_score(self, scores, word):
        if not self.lemmatize:
            return scores[word]
        else:
            lemmas = self.wordsToLemmas[word] if self.lemmatize else [word]
            return max([scores[lemma] for lemma in lemmas if lemma in scores])

    # TODO (Daryl): Look into interleaving nouns/adjs with adverbs and other POS
    def combine_to_keyphrases(self, taggedTokens, scores, min_num_labels):
        combinedKeyphraseScores = {}
        sortedScores = sorted(scores.items(), key=lambda x:x[1], reverse=True)[:int(self.keywordThreshold*min_num_labels)]
        
        # Construct keyword set
        if self.lemmatize:
            keywords = set([word for lemma, score in sortedScores for word in self.lemmasToWords[lemma]])
        else:
            keywords = set([word for word, score in sortedScores])

        keyphraseScores = {}
    
        # Combine keywords into keyphrases
        keyphrase = ()
        for word, tag in self.taggedWords:
            if word in keywords:
                keyphrase += (word,)
            else:
                candidateKeyphrases = cooccurrence.findNgrams(keyphrase)
                bestKeyphrase, bestKeyphraseScore = None, None
                for candidateKeyphrase in candidateKeyphrases:
                    if candidateKeyphrase and ' '.join(candidateKeyphrase) in self.text:
                        score = sum([self.get_best_score(scores, keyword) for keyword in candidateKeyphrase])
                        if self.lengthPenaltyFn:
                            # print 'Length penalty: %s being reduced from %s for length of %s' % (self.lengthPenaltyFn(len(keyphrase)), score, len(keyphrase))
                            score -= self.lengthPenaltyFn(len(candidateKeyphrase))
                        if score > bestKeyphraseScore:
                            bestKeyphrase, bestKeyphraseScore = candidateKeyphrase, score
                if bestKeyphrase:
                    keyphraseScores[bestKeyphrase] = bestKeyphraseScore
                    keyphrase = ()

        if self.useNgrams:
            self.addCommonNgramsAndScores(keywords, scores, keyphraseScores)

        # Add this set of keyphrase scores to the combined score dict,
        # resolving conflicts by taking the higher score.
        for keyphrase in keyphraseScores:
            combinedKeyphraseScores[keyphrase] = max(
                combinedKeyphraseScores[keyphrase],
                keyphraseScores[keyphrase],
            ) if keyphrase in combinedKeyphraseScores else keyphraseScores[keyphrase]

        keyphrases = sorted(combinedKeyphraseScores.keys(), key=combinedKeyphraseScores.get, reverse=True)
        result = [' '.join(keyphrase) for keyphrase in keyphrases][:min_num_labels+self.numExtraLabels]

        # If we haven't reached the minimum number of labels, add unigrams in order of score
        if self.lemmatize:
            unigrams_and_scores = [(word, score) for lemma, score in sortedScores for word in self.lemmasToWords[lemma]]
        else:
            unigrams_and_scores = [(word, score) for word, score in sortedScores]
        unigrams_and_scores = sorted(unigrams_and_scores, key=lambda t: t[1], reverse=True)
        i = 0
        while i < len(unigrams_and_scores) and \
            len(result) < min_num_labels + self.numExtraLabels and \
            unigrams_and_scores[i][0] not in result:
            result.append(unigrams_and_scores[i][0])
            i += 1
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
