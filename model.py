import random

import torch
from torch import nn

import auxiliary_functions as af
from config import DEVICE, DROPOUT, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS


# Слои - 1. Embedding; 2. LSTM (двунаправленный); 3. Linear (переводит из двунаправленного в однонаправленный)
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # каждый индекс слова переводит в вектор размерности embedding_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_len): # src_len - тензор с длинами слов в пртедложениях
        embedded = self.dropout(self.embedding(src)) # (batch_size, src_len, embedding_dim) - размер тензора
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len, batch_first=True, enforce_sorted=False) # убираем лишние паддинги, чтобы LSTM не учитывал их при обучении
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) # (batch_size, src_len, hidden_dim * 2) - outputs тензор с выходными данными LSTM
        hidden = torch.tanh(self.fc_hidden(torch.cat((hidden[-2], hidden[-1]), dim=1))) # hidden[-2] - последний слой прямое направление, hidden[-1] - последний слой обратное направление, cat - склеивает два тензора
        cell = torch.tanh(self.fc_cell(torch.cat((cell[-2], cell[-1]), dim=1))) # cell[-1] -  последний слой обратное направление, cell[-2] - последний слой прямое направление
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1) # (num_layers, batch_size, hidden_dim) - 2 - потому что во внимании нужно 2 слоя, в декодере тоже 2
        cell = cell.unsqueeze(0).repeat(self.num_layers, 1, 1) # (num_layers, batch_size, hidden_dim) - 2 - потому что в декодер надо будет 2 слоя
        return outputs, hidden, cell

# класс внимания: мы берем нынешний слой декодера и сравниваем его с каждым элементом encoder_outputs, после мы вычисляем энергию внимания и перемножаем ее с encoder_outputs.
# тем самым мы получаем контекст, который показывает насколько каждый элемент encoder_outputs соответствует текущему слою декодера
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False) # bias=False - не используем смещение

    def forward(self, decoder_hidden, encoder_outputs, mask):
        _, src_len, _ = encoder_outputs.shape
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1) # расширяем decoder_hidden для выравнивания с каждым элементом encoder_outputs
        combined = torch.cat((decoder_hidden, encoder_outputs), dim=2) # объединяем decoder_hidden и encoder_outputs для вычисления энергии внимания
        energy = torch.tanh(self.attn(combined)) # вычисляем энергию внимания (tanh - активационная функция от -1 до 1)
        attention = self.v(energy).squeeze(2) # получаем вектор из оценок для каждого слова
        attention = attention.masked_fill(mask == 0, -1e10) # маскируем паддинг, чтобы не придавало им важности
        attention_weights = torch.softmax(attention, dim=1) # превращаем оценки в вероятность
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1) # bmm - перемножение матриц
        return context, attention_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim * 2, hidden_dim, num_layers, batch_first=True, dropout=dropout) # batch_first=True - первым измерением будет размер батча
        self.fc = nn.Linear(hidden_dim * 3, output_dim)

    def forward(self, input, encoder_outputs, decoder_hidden, decoder_cell, mask):
        embedded = self.dropout(self.embedding(input))
        context, attention_weight = self.attention(decoder_hidden[-1], encoder_outputs, mask)
        context = context.unsqueeze(1)
        lstm_input = torch.cat((embedded, context), dim=2) # склеиваем по 2 размерности (очень важно) - (batch_size, 1, embedding_dim + hidden_dim * 2)
        output, (decoder_hidden, decoder_cell) = self.lstm(
        lstm_input, (decoder_hidden, decoder_cell) # (hidden, cell) - начальные состояния LSTM, то есть  конечные у encoder
        )
        output, context = output.squeeze(1), context.squeeze(1)
        combined = torch.cat((output, context), dim=1)
        predictions = self.fc(combined)
        return predictions, decoder_hidden, decoder_cell, attention_weight



class TranslationModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder = Encoder(
            src_vocab_size,
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout
        )

        self.decoder = Decoder(
            tgt_vocab_size,
            embedding_dim,
            hidden_dim,
            tgt_vocab_size,
            num_layers,
            dropout
        )

        # ID специальных токенов (будут установлены позже)
        self.pad_id = None
        self.bos_id = None
        self.eos_id = None

    def set_special_tokens(self, tgt_vocab):
        self.pad_id = tgt_vocab.pad_id
        self.bos_id = tgt_vocab.bos_id
        self.eos_id = tgt_vocab.eos_id

    def forward(self, src, tgt, src_len, teacher_forcing_ratio=0.5):
        if self.pad_id is None or self.bos_id is None or self.eos_id is None:
            raise ValueError("Special tokens are not set")
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        encoder_outputs, hidden, cell = self.encoder(src, src_len) # мы вызываем encoder и вызываем __call__, который в Pytorch вызывает forward
        src_mask = (src != self.pad_id)
        decoder_input = tgt[:, 0].unsqueeze(1) # делаем размер (batch_size, 1)
        outputs = torch.zeros(batch_size, tgt_len, self.decoder.output_dim, device=src.device)

        for t in range(1, tgt_len):
            decoder_output, hidden, cell, _ = self.decoder(
                decoder_input,
                encoder_outputs,
                hidden,
                cell,
                src_mask
            ) # мы вызываем decoder и вызываем __call__, который в Pytorch вызывает forward
            outputs[:, t, :] = decoder_output # мы вставляем в тензор по слову в каждое предложение батча
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = tgt[:, t].unsqueeze(1) # правильный токен из target
            else:
                decoder_input = decoder_output.argmax(1).unsqueeze(1)  # предсказание модели

        return outputs

    @torch.no_grad()
    def translate(self, source_texts, src_vocab, tgt_vocab, device, beam_size, max_len=30):
        self.set_special_tokens(tgt_vocab)

        self.eval()
        self.to(device)

        translated_texts = []

        for source_text in source_texts:
            tokens = af.tokenize(source_text)
            src_ids = [src_vocab.bos_id] + src_vocab.encode(tokens) + [src_vocab.eos_id]
            src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
            src_len = torch.tensor([len(src_ids)], dtype=torch.long)

            src_mask = (src_tensor != src_vocab.pad_id)
            encoder_outputs, hidden, cell = self.encoder(src_tensor, src_len)
            beams = [(0.0, [tgt_vocab.bos_id], hidden, cell)] # Создаем список с бимами (beam_size лучших вариантов из всех получившихся)
            finished = [] # сюда будем складывать лучшие beam_size вариантов предложений

            for _ in range(max_len):
                new_beams = [] # сюда перезаписываем новые
                for score, ids, h, c in beams:
                    # Проверяем на то, стоит ли последний индекс - индекс окончания предложения. Если так, то предложение закончено и продолжаем, если нет, то дальше идет вниз.
                    if ids[-1] == tgt_vocab.eos_id:
                        finished.append((score, ids))
                        continue
                    dec_in = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)
                    out, h2, c2, _ = self.decoder(dec_in, encoder_outputs, h, c, src_mask)
                    logp = torch.log_softmax(out, dim=-1).squeeze(0)  # (vocab_size,) - делаем логированные вероятности
                    topk = torch.topk(logp, beam_size) # получаем на выходе словарь индекс из списка (место, на котором был в списке) и score - значение логированной вероятности
                    # перебираем варианты для для каждого кандидата и выбираем лучших beam_size
                    for k in range(beam_size):
                        new_beams.append((
                            score + topk.values[k].item(),
                            ids + [topk.indices[k].item()],
                            h2, c2,
                        ))
                if not new_beams:
                    break
                new_beams.sort(key=lambda x: x[0], reverse=True)
                beams = new_beams[:beam_size]

            finished.extend([(s, ids) for s, ids, _, _ in beams]) # expend - добавляет в конец список все значения списка,  который передали
            best_ids = max(finished, key=lambda x: x[0])[1][1:] # max(1, ...) - чтобы перестраховаться с делением на 0, а делим потому что в предложение столько слов
            translated_texts.append(" ".join(tgt_vocab.decode(best_ids))) # добавляем строчку
        return translated_texts


# Функция для загрузки модели
def load_model(path, src_vocab_size, tgt_vocab_size, tgt_vocab, device = DEVICE):
    model = TranslationModel(
        src_vocab_size, tgt_vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    model.set_special_tokens(tgt_vocab)
    model.load_state_dict(torch.load(path, map_location=device)) # map_location=device - куда ложит веса на cpu или gpu
    model.to(device)
    model.eval()
    return model
