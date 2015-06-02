from parse import handwritten_data_reader, inspec_data_reader, duc_data_reader

DATASETS = set([
    'Handwritten',
    'Inspec',
    'DUC-2001',
])
READERS = {
    'Handwritten': handwritten_data_reader,
    'Inspec': inspec_data_reader,
    'DUC-2001': duc_data_reader,
}
MISTAKES_FILENAME = 'mistakes.txt'
SKIP_DATASETS = ['Handwritten']
MODEL_KEYWORD = 'model'