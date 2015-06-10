from parse import handwritten_data_reader, inspec_data_reader, duc_data_reader

import nltk
from nltk.corpus import wordnet
from subprocess import *


def convert_filename_to_pos_tagged(filename):
    return filename.replace('data/', 'data/POS-Tagged/')

# Maps from NLTK POS tags to WordNet POS tags
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
        return ''

def tag(tokens):
    words = ' '.join(tokens).encode('utf-8').strip() # Avoid ascii codec error
    command = "apache-opennlp/bin/opennlp POSTagger apache-opennlp/models/en-pos-perceptron.bin"
    p = Popen(command, shell=True, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    output = p.communicate(input=words)[0].split('\n')[1]
    result = []
    for token in output.split():
        if token.endswith('_``'):
            p = Popen(command, shell=True, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
            token = p.communicate(input=token.split('_')[0])[0].split('\n')[1]
        if token:
            tag_index = token.rfind('_')
            word, tag = token[:tag_index], token[tag_index+1:]
            result.append(word + '_' + (wordnetPosCode(tag)))
    return result

def get_tokens(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    return tokens

def get_output_text(tagged_tokens):
    return ' '.join(tagged_tokens).encode('utf-8').strip()

def write_tokens_to_file(tagged_tokens, filename):
    out_text = get_output_text(tagged_tokens)
    with open(filename, 'w') as f:
        f.write(out_text)

def convert_examples(examples):
    num_done = 0
    for filename, text, labels in examples:
        tagged_tokens = tag(get_tokens(text))
        write_tokens_to_file(tagged_tokens, convert_filename_to_pos_tagged(filename))
        num_done += 1
        if num_done % 10 == 0:
            print 'Finished %s of %s examples' % (num_done, len(examples))

if __name__=='__main__':
    #examples = inspec_data_reader()[250:]
    examples = duc_data_reader()[20:40]
    convert_examples(examples)
