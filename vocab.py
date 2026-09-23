import pickle

from config import (
    BOS,
    EOS,
    PAD,
    UNK,
    VOCABS_PATH,
)


# Класс для создания словаря,  в котором хранятся все символы из выборки
class Vocab:
    def __init__(self, token_lists, min_freq: int = 1, specials=None):
        if specials is None:
            specials = [PAD, BOS, EOS, UNK]
        # freq - словарь частотности токенов
        freq = {}
        for tokens in token_lists:
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
        # itos - список токенов, с минимальной частотностью min_freq, отсортированных по частотности
        self.itos = list(specials)
        for token, count in sorted(freq.items()):
            if count >= min_freq and token not in self.itos:
                self.itos.append(token)
        # stoi - словарь индексов токенов, отсортированных по частотности
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    # encode - метод для кодирования списка токенов в список индексов
    def encode(self, tokens):
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    # decode - метод для декодирования списка индексов в список токенов
    # ids - список индексов, skip_specials - флаг для пропуска специальных токенов
    def decode(self, ids, skip_specials=True):
        tokens = []
        specials = {PAD, BOS, EOS}
        for idx in ids:
            token = self.itos[int(idx)]
            if skip_specials and token in specials:
                continue
            tokens.append(token)
        return tokens

    def __len__(self):
        return len(self.itos)


# Функция загрузки словаря из файла
def load_vocab(path = VOCABS_PATH):
    with open(path, "rb") as f: # rb - открывает файл в бинарном режиме
        vocabs = pickle.load(f)
        return vocabs["ru"], vocabs["en"]
