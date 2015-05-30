import evaluate
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer

class BaseModel:

    def __init__(self, stripPunct=True, stemRule=None, lemmatize=False, stripStopWords=True, synFilter=None):
        self.stripPunct = stripPunct
        self.stemRule = stemRule
        self.lemmatize = lemmatize
        self.stripStopWords = stripStopWords
        self.synFilter = synFilter

    def tokenize(self, text):
        # Strip punctuation if unneeded for co-occurrence counts
        if self.stripPunct:
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(text)
        else:
            tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

        # Normalize words
        words = [t.lower() for t in tokens]
        
        # Stemming. Faster than lemmatization but imprecise
        if self.stemRule:
            stemmer = nltk.PorterStemmer() if self.stemRule=='Porter' else nltk.LancasterStemmer()
            words = [stemmer.stem(w) for w in words]

        # Part of speech tagging
        if self.lemmatize or self.synFilter:
        	self.taggedWords = nltk.pos_tag(words)

        # Lemmatization
        if self.lemmatize:
            lemmatizer = nltk.WordNetLemmatizer()
            taggedWords = nltk.pos_tag(words)
            words = [lemmatizer.lemmatize(word, self.wordnetPosCode(tag)) for word, tag in taggedWords]

        return words

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

    def extract_keywords(self, text, min_num_labels):
        raise NotImplementedError

    def evaluate(self):
        evaluate.evaluate_extractor(self.extract_keywords)
