"""Настройки проекта: устройство, пути к файлам, гиперпараметры модели."""

import torch

# Устройство
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Пути к файлам
RU2EN_MODEL_PATH = "data/ru2en_model.pth"
EN2RU_MODEL_PATH = "data/en2ru_model.pth"
VOCABS_PATH = "data/vocabs.pkl"

#  Гиперпараметры модели (Совпадают с теми, что использовались при обучении)
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2

# Параметры для translate
BEAM_SIZE = 1
MAX_LEN = 30

# Специальные токены для словаря
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
