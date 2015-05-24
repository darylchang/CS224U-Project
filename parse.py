import nltk
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer

ENCODING = 'utf-8'

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

def tokenize(text, stripPunct=True, stemRule=None, lemmatize=False):
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

def clean_label(label):
    return ' '.join(label.strip().lower().split())

def handwritten_data_reader(**kwargs):
    # Simple data format for handwritten document/keyword examples. The first
    # line of each file listed in handwritten_docs.txt is taken to be 
    data_dir = 'data/Handwritten/'
    with open(data_dir + 'handwritten_docs.txt', 'r') as docs_file:
        filenames = [filename.strip() for filename in docs_file.readlines()]

    examples = []
    for filename in filenames:
        with open(data_dir + filename, 'r') as f:
            label_line = f.readline().decode(ENCODING)
            text = f.read().decode(ENCODING)
        labels = set([clean_label(label) for label in label_line.split(';') if label])
        examples.append((tokenize(text, **kwargs), labels))
    return examples

def inspec_data_reader(**kwargs):
    data_dir = 'data/Inspec/'
    with open(data_dir + 'test_abstracts.txt', 'r') as abstracts_list:
        abstracts_filenames = [filename.strip() for filename in abstracts_list.readlines()]
    with open(data_dir + 'test_labels.txt', 'r') as labels_list:
        labels_filenames = [filename.strip() for filename in labels_list.readlines()]

    examples = []
    for abstract_file, labels_file in zip(abstracts_filenames, labels_filenames):
        with open(data_dir + abstract_file, 'r') as f:
            text = f.read().decode(ENCODING)
        with open(data_dir + labels_file, 'r') as f:
            label_text = f.read().decode(ENCODING)
        labels = set([clean_label(label) for label in label_text.strip().split(';') if label])
        examples.append((tokenize(text, **kwargs), labels))

    return examples

def duc_data_reader(**kwargs):

    examples = []

    return examples

if __name__=='__main__':
    # print handwritten_data_reader(lemmatize=False)
    print inspec_data_reader()[:1]
