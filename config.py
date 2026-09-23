"""Настройки проекта: устройство, пути к файлам, гиперпараметры модели."""

import torch

# Устройство
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Пути к файлам
def get_model_paths():
    while True:
        a = input("Введите 'easy' или 'hard' в зависимости от уровня сложности от модели, которую хотите использовать: ")
        if a == "easy":
            print("Загрузка easy model...")
            return "data_easy/ru2en_model.pth", "data_easy/en2ru_model.pth", "data_easy/vocabs.pkl"
        elif a == "hard":
            print("Загрузка hard model...")
            return "data_hard/ru2en_model.pth", "data_hard/en2ru_model.pth", "data_hard/vocabs.pkl"
        elif a == "myself":
            return "data/ru2en_model.pth", "data/en2ru_model.pth", "data/vocabs.pkl"
        else:
            print("Неверный ввод. Пожалуйста, введите 'easy' или 'hard':")

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
