from parse import handwritten_data_reader, inspec_data_reader, duc_data_reader

HANDWRITTEN_DATASET = 'Handwritten'
INSPEC_DATASET = 'Inspec'
DUC_DATASET = 'DUC-2001'

DATASETS = set([
    HANDWRITTEN_DATASET,
    INSPEC_DATASET,
    DUC_DATASET,
])
READERS = {
    HANDWRITTEN_DATASET: handwritten_data_reader,
    INSPEC_DATASET: inspec_data_reader,
    DUC_DATASET: duc_data_reader,
}
MISTAKES_FILENAME = 'mistakes.txt'
MODEL_KEYWORD = 'model'