from config import EN2RU_MODEL_PATH, RU2EN_MODEL_PATH
from model import load_model
from TranslationApp import TranslationApp
from vocab import load_vocab

# Загружаем модели и словари
ru_vocab, en_vocab = load_vocab()
print("Словари загружены")

ru_vocab_size = len(ru_vocab)
en_vocab_size = len(en_vocab)

ru2en_model = load_model(RU2EN_MODEL_PATH, ru_vocab_size, en_vocab_size, tgt_vocab=en_vocab)
en2ru_model = load_model(EN2RU_MODEL_PATH, en_vocab_size, ru_vocab_size, tgt_vocab=ru_vocab)
print("Модели загружены")

# Запускаем приложение
app = TranslationApp(ru2en_model, en2ru_model, ru_vocab, en_vocab)
app.run()
print("Приложение запущено")
