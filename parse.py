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

def get_semicolon_separated_labels(label_text):
    return set([clean_label(label) for label in label_text.strip().split(';') if label])

def handwritten_data_reader():
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
        labels = get_semicolon_separated_labels(label_line)
        examples.append((data_dir + filename, text, labels))
    return examples

def inspec_data_reader():
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
        labels = get_semicolon_separated_labels(label_text)
        examples.append((data_dir + abstract_file, text, labels))
    return examples

def process_duc_labels(labels_lines):
    labels = {}
    for line in labels_lines:
        document, label_text = line.split('@')
        labels[document] = get_semicolon_separated_labels(label_text)
    return labels

def process_duc_text(text):
    # Following usage in literature, only consider text in DUC-2001 dataset
    # articles between the <TEXT> tags.
    return text[text.find("<TEXT>") + len("<TEXT>") : text.rfind("</TEXT>")]

def duc_data_reader():
    data_dir = 'data/DUC2001/'
    with open(data_dir + 'documents.txt') as documents_list:
        document_filenames = [filename.strip() for filename in documents_list.readlines()]
    with open(data_dir + 'annotations.txt', 'r') as labels_file:
        labels_lines = [line.decode(ENCODING).strip() for line in labels_file.readlines()]
    
    examples = []
    labels = process_duc_labels(labels_lines)
    for document_filename in document_filenames:
        with open(data_dir + document_filename, 'r') as document:
            text = document.read().decode(ENCODING)
        article_text = process_duc_text(text)
        examples.append((data_dir + document_filename, article_text, labels[document_filename]))
    return examples

if __name__=='__main__':
    # print handwritten_data_reader(lemmatize=False)
    # print inspec_data_reader()[:1]
    print duc_data_reader()[0]
