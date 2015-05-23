import nltk
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer

############ Parse.py ####################
# Parses text documents into a list of tokens.
# Handles multiple file formats.

def parse(filename, stripPunct=True, stemRule=None, lemmatize=True):
	with open (filename, 'r') as f:
		text = f.read()

		# Strip punctuation if unneeded for co-occurrence counts
		if stripPunct:
			tokenizer = RegexpTokenizer(r'\w+')
			tokens = tokenizer.tokenize(text)
		else:
			tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

		# Normalize words
		words = [t.lower() for t in tokens]
		
		# Stemming. Faster than lemmatization but imprecise
		if stemRule:
			stemmer = nltk.PorterStemmer() if stemRule=='Porter' else nltk.LancasterStemmer()
			words = [stemmer.stem(w) for w in words]

		# Lemmatization.
		if lemmatize:
			lemmatizer = nltk.WordNetLemmatizer()
			taggedWords = nltk.pos_tag(words)
			words = [lemmatizer.lemmatize(word, wordnetPosCode(tag)) for word, tag in taggedWords]

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

def main():
	parse('data/litReview.txt', lemmatize=True)

if __name__=='__main__':
	main()