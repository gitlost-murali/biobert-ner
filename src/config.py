import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = "../input/biobert-large-cased-v1.1"
MODEL_PATH = "biomodel_on_custom.bin"
BASE_MODEL_DIM = 1024 # 1024 for bio & 768 for BERT
TRAINING_FILE = "../input/train.tsv" # Use the function process_data_conll
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=False
)
