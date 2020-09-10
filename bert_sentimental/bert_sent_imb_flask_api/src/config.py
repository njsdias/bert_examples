import transformers
import torch

#DEVICE = "cuda"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "../input/bert_base_uncased/"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../input/imbd.csv"
TOKENIZER = transformers.BertTokenizer(BERT_PATH, do_lower_case=True)


