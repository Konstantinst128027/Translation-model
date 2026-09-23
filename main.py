from config import get_model_paths
from model import load_model
from TranslationApp import TranslationApp
from vocab import load_vocab

# Выбираем пути к моделям и словарям
ru2en_model_path, en2ru_model_path, vocabs_path = get_model_paths()

# Загружаем модели и словари
ru_vocab, en_vocab = load_vocab(vocabs_path)
print("Словари загружены")

ru_vocab_size = len(ru_vocab)
en_vocab_size = len(en_vocab)

ru2en_model = load_model(ru2en_model_path, ru_vocab_size, en_vocab_size, tgt_vocab=en_vocab)
en2ru_model = load_model(en2ru_model_path, en_vocab_size, ru_vocab_size, tgt_vocab=ru_vocab)
print("Модели загружены")

# Запускаем приложение
app = TranslationApp(ru2en_model, en2ru_model, ru_vocab, en_vocab)
app.run()
print("Приложение запущено")
