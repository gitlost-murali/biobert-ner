import transformers
from transformers import AutoTokenizer

MAX_LEN = 128
TRAIN_BATCH_SIZE = 12
VALID_BATCH_SIZE = 4
EPOCHS = 10
CUDA = True
BASE_MODEL_PATH = "../input/Bio_ClinicalBERT" #bert-base-uncased OR biobert-large-cased-v1.1
MODEL_PATH = "model.bin" #biomodel_on_custom.bin OR model.bin
BASE_MODEL_DIM = 768 # 1024 for bioBERT & 768 for BERT
TRAINING_FILE = "../../../datasets/experiment_data_en/english_protein_chemical.json" # Use the function process_data_conll()
# TRAINING_FILE = "../input/dldata_bilou_7k.json" # Use the function read_bilou
TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
# TOKENIZER = transformers.BertTokenizer.from_pretrained(
#     BASE_MODEL_PATH,
#     do_lower_case=False #remember to change based on the model.
# )
RANDOM_STATE = 42 #Use this for consistent shuffle/splits.