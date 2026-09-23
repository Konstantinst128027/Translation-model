import random
import re

import numpy as np
import torch


# Функция для установки случайного состояния
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str):
    text = text.lower()  # приводим к нижнему регистру
    # Вставляем пробел перед и после каждого знака препинания
    # Знаки: , . ! ? ; : ( ) [ ] { } « » " ' ` - – — / \ | < > @ # $ % ^ & * + = ~ _
    text = re.sub(r'([,.!?;:()\[\]{}«»"\'`\-–—/\\|<>@#$%^&*+=~_])', r' \1 ', text)
    # Удаляем всё, кроме букв, цифр, пробелов и уже вставленных знаков (уже обработано)
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    # Схлопываем пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()


# Функция, которая делает batch из данных множества, делает padding (все строки выравниет под один размер, заполняя их src_pad_id)
# На выходе получаем словарь с тензорами src, src_len и source_text и если tgt_pad_id не None, то также tgt, tgt_len и target_text
def collate_batch(batch, src_pad_id, tgt_pad_id):
    src_lens = [len(x["src_ids"]) for x in batch]
    max_src_len = max(src_lens)
    src = torch.full((len(batch), max_src_len), src_pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        src[i, :len(x["src_ids"])] = x["src_ids"]
    result = {
        'src': src,
        'src_len': torch.tensor(src_lens, dtype=torch.long),
        'source_text': [x['source_text'] for x in batch]
    }

    tgt_lens = [len(x["tgt_ids"]) for x in batch]
    max_tgt_len = max(tgt_lens)
    tgt = torch.full((len(batch), max_tgt_len), tgt_pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        tgt[i, :len(x["tgt_ids"])] = x["tgt_ids"]
    result["tgt"] = tgt
    result["tgt_len"] = torch.tensor(tgt_lens, dtype=torch.long)
    result["target_text"] = [x['target_text'] for x in batch]

    return result
