import networkx as nx
import cooccurrence
import evaluate
from parse import tokenize

class BaseModel:

	def __init__(self, stripPunct=True, stemRule=None, lemmatize=False):
		self.stripPunct = stripPunct
		self.stemRule = stemRule
		self.lemmatize = lemmatize

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

    # Lemmatization
    if self.lemmatize:
        lemmatizer = nltk.WordNetLemmatizer()
        taggedWords = nltk.pos_tag(words)
        words = [lemmatizer.lemmatize(word, self.wordnetPosCode(tag)) for word, tag in taggedWords]

    return words

    # Maps from NLTK POS tags to WordNet POS tagsC
	def wordnetPosCode(tag):
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
