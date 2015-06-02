import cooccurrence
import evaluate
import networkx as nx
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
import numpy as np


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

    def tokenize(self, text):
        # Strip punctuation if unneeded for co-occurrence counts
        if self.stripPunct:
            pattern = r'''(?x)           # set flag to allow verbose regexps
                      ([A-Z]\.)+         # abbreviations, e.g. U.S.A.
                      | \-?[\d\w]+([-']\w+)*    # words w/ optional internal hyphens/apostrophe
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
            taggedWords = [(word, self.wordnetPosCode(tag)) for word, tag in nltk.pos_tag(words)]

        # Lemmatize
        if self.lemmatize:
        	lemmatizer = nltk.WordNetLemmatizer()
        	words = [lemmatizer.lemmatize(word, tag) for word, tag in taggedWords]

        return taggedWords if self.synFilter else words

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
            return wordnet.NOUN

    def create_graph(self, words):
        cooccurrenceDict = cooccurrence.slidingWindowMatrix(words, self.windowSize, self.synFilter, self.stripStopWords)
        return nx.Graph(cooccurrenceDict)

    # TODO (Daryl): Look into interleaving nouns/adjs with adverbs and other POS
    # TODO (all): try allowing combinations of nonadjacent keywords using frequent n-grams
    # TODO (all): try model w/ ensemble approach for final keyphrase scoring
    def combine_to_keyphrases(self, text, words, scores_list, min_num_labels):
        results = []
        for scores in scores_list:
            sorted_scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)[:self.keywordThreshold*min_num_labels]
            keywords = [keyword for keyword, score in sorted_scores]
            keyphrases = set([(keyword,) for keyword in keywords])
            keyphrase_scores = {(keyword,): scores[keyword]-self.lengthPenaltyFn(1) for keyword in keywords}

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
                            score = np.sum([scores[keyword] for keyword in candidateKeyphrase])
                            if self.lengthPenaltyFn:
                                #print 'Length penalty: %s being reduced from %s for length of %s' % (self.lengthPenaltyFn(len(keyphrase)), score, len(keyphrase))
                                score -= self.lengthPenaltyFn(len(candidateKeyphrase))
                            keyphrase_scores[candidateKeyphrase] = score
                            keyphrases.add(candidateKeyphrase)
                    keyphrase = ()

            keyphrases = sorted(keyphrases, key=keyphrase_scores.get, reverse=True)
            results += [' '.join(keyphrase) for keyphrase in keyphrases][:min_num_labels+self.numExtraLabels]
        return results

    def extract_keyphrases(self, text, min_num_labels):
        raise NotImplementedError

    def evaluate(self, numExamples=None, compute_mistakes=False, verbose=False, skip_datasets=[]):
        return evaluate.evaluate_extractor(self.extract_keyphrases, numExamples, compute_mistakes, verbose, skip_datasets)
