import nltk
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer

#################### Parse.py ####################
# Contains readers that convert the various data sets into lists of
# (tokens, keywords) pairs.

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

def tokenize(text, stripPunct=True, stemRule=None, lemmatize=True):
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

def handwritten_data_reader(**kwargs):
    # Simple data format for handwritten document/keyword examples. The first
    # line of each file listed in handwritten_docs.txt is taken to be 
    data_dir = 'data/Handwritten/'
    with open(data_dir + 'handwritten_docs.txt', 'r') as docs_file:
        filenames = [filename.strip() for filename in docs_file.readlines()]

    examples = []
    for filename in filenames:
        with open(data_dir + filename, 'r') as f:
            label_line = f.readline().decode('utf-8')
            text = f.read().decode('utf-8')
        labels = set([label.strip().lower() for label in label_line.split(';')])
        examples.append((tokenize(text, **kwargs), labels))
    return examples

if __name__=='__main__':
    print handwritten_data_reader(lemmatize=False)
