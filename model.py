import evaluate
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
import nltk

class BaseModel:

    def __init__(self, stripPunct=True, stemRule=None, lemmatize=False, 
    			 stripStopWords=True, synFilter=[wordnet.NOUN, wordnet.ADJ], 
                 windowSize=None, keywordThreshold=5):
        self.stripPunct = stripPunct
        self.stemRule = stemRule
        self.lemmatize = lemmatize
        self.stripStopWords = stripStopWords
        self.synFilter = synFilter
        self.windowSize = windowSize
        self.keywordThreshold = keywordThreshold

    def tokenize(self, text):
        # Strip punctuation if unneeded for co-occurrence counts
        if self.stripPunct:
            tokenizer = RegexpTokenizer(r'[\w\-]+')
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

    def extract_keyphrases(self, text, min_num_labels):
        raise NotImplementedError

    def evaluate(self, numExamples=None, verbose=False):
        return evaluate.evaluate_extractor(self.extract_keyphrases, numExamples, verbose)
