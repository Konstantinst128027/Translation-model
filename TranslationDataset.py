import torch
from torch.utils.data.dataset import Dataset

import auxiliary_functions as af


# Наследуется от Dataset, чтобы использовать его в DataLoader
class TranslationDataset(Dataset):
    def __init__(self, df, src_vocab, tgt_vocab , reverse = False):
        self.df = df.reset_index(drop=True)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.reverse = reverse

    def __len__(self):
        return len(self.df)

    # Получение элемента по индексу, получаем строчку по индексу и преобразуем в словарь
    def __getitem__(self, idx):
        if self.reverse:
            row = self.df.iloc[idx]
            src_text = row['target']
            tgt_text = row['source']
        else:
            row = self.df.iloc[idx]
            src_text = row['source']
            tgt_text = row['target']
        src_ids = [self.src_vocab.bos_id] + self.src_vocab.encode(af.tokenize(src_text)) + [self.src_vocab.eos_id]
        tgt_ids = [self.tgt_vocab.bos_id] + self.tgt_vocab.encode(af.tokenize(tgt_text)) + [self.tgt_vocab.eos_id]
        item = {
            'source_text': src_text,
            'src_ids': torch.tensor(src_ids, dtype=torch.long), # dtype - long, потому что целые числа должны быть у индексов
            'target_text': tgt_text,
            'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long)
        }


        return item
