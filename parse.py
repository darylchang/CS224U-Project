import nltk
import re


ENCODING = 'utf-8'
USE_ALL_INSPEC = False


#################### Parse.py ####################
# Contains readers that convert the various data sets into lists of
# (tokens, keywords) pairs.

def clean_text(label):
    return ' '.join([token for token in label.strip().lower().split() if token])

def get_semicolon_separated_labels(label_text):
    return set([clean_text(label) for label in label_text.strip().split(';') if label])

def handwritten_data_reader():
    # Simple data format for handwritten document/keyword examples. The first
    # line of each file listed in handwritten_docs.txt is taken to be 
    data_dir = 'data/Handwritten/'
    with open(data_dir + 'handwritten_docs.txt', 'r') as docs_file:
        filenames = [filename.strip() for filename in docs_file.readlines() if not filename.startswith('#')]

    examples = []
    for filename in filenames:
        with open(data_dir + filename, 'r') as f:
            label_line = f.readline().decode(ENCODING)
            text = clean_text(f.read().decode(ENCODING))
        labels = get_semicolon_separated_labels(label_line)
        examples.append((data_dir + filename, text, labels))
    return examples

def inspec_data_reader(use_pretagged=True):
    data_dir = 'data/Inspec/'
    text_data_dir = 'data/POS-Tagged/Inspec/' if use_pretagged else data_dir
    abstracts_listings = ['test_abstracts.txt']
    labels_listings = ['test_labels.txt']
    if USE_ALL_INSPEC:
        abstracts_listings += ['train_abstracts.txt', 'dev_abstracts.txt']
        labels_listings += ['train_labels.txt', 'dev_labels.txt']
    abstracts_filenames, labels_filenames = [], []
    for abstracts_listing in abstracts_listings:
        with open(data_dir + abstracts_listing, 'r') as abstracts_list:
            abstracts_filenames += [filename.strip() for filename in abstracts_list.readlines()]
    for labels_listing in labels_listings:
        with open(data_dir + labels_listing, 'r') as labels_list:
            labels_filenames += [filename.strip() for filename in labels_list.readlines()]

    examples = []
    for abstract_file, labels_file in zip(abstracts_filenames, labels_filenames):
        with open(text_data_dir + abstract_file, 'r') as f:
            raw_text = f.read().decode(ENCODING)
            raw_lines = raw_text.strip().split('\r\n')
            text = '\n'.join([clean_text(line) for line in raw_lines])
        with open(data_dir + labels_file, 'r') as f:
            label_text = f.read().decode(ENCODING)
        labels = get_semicolon_separated_labels(label_text)
        examples.append((text_data_dir + abstract_file, text, labels))
    return examples

def process_duc_labels(labels_lines):
    labels = {}
    for line in labels_lines:
        document, label_text = line.split('@')
        labels[document] = get_semicolon_separated_labels(label_text)
    return labels

def process_duc_text(text):
    # Following usage in literature, only consider text in DUC-2001 dataset
    # articles between the <TEXT> tags. In addition, throw out the <P> tags
    # present in the LA articles.
    text = text[text.find("<TEXT>") + len("<TEXT>") : text.rfind("</TEXT>")]
    return re.sub(r'</?P>', '', text)

def duc_data_reader(use_pretagged=True):
    data_dir = 'data/DUC2001/'
    text_data_dir = 'data/POS-Tagged/DUC2001/' if use_pretagged else data_dir  
    with open(data_dir + 'documents.txt') as documents_list:
        document_filenames = [filename.strip() for filename in documents_list.readlines()]
    with open(data_dir + 'annotations.txt', 'r') as labels_file:
        labels_lines = [line.decode(ENCODING).strip() for line in labels_file.readlines()]
    
    examples = []
    labels = process_duc_labels(labels_lines)
    for document_filename in document_filenames:
        with open(text_data_dir + document_filename, 'r') as document:
            text = document.read().decode(ENCODING)
        article_text = clean_text(process_duc_text(text))
        examples.append((text_data_dir + document_filename, article_text, labels[document_filename]))
    return examples

if __name__=='__main__':
    # print handwritten_data_reader(lemmatize=False)
    # print inspec_data_reader()[:1]
    print duc_data_reader()[0]
