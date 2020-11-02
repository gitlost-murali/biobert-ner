import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 2
EPOCHS = 10
CUDA = False
BASE_MODEL_PATH = "../input/biobert-large-cased-v1.1" #bert-base-uncased OR biobert-large-cased-v1.1
MODEL_PATH = "biomodel_on_ade.bin" #biomodel_on_custom.bin OR model.bin
BASE_MODEL_DIM = 1024 # 1024 for bio & 768 for BERT
# TRAINING_FILE = "../input/train.tsv" # Use the function process_data_conll
TRAINING_FILE = "../input/dldata_bilou_7k.json" # Use the function read_bilou
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=False
)