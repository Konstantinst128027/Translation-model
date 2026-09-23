import tkinter as tk
from tkinter import messagebox

from config import BEAM_SIZE, DEVICE, MAX_LEN

BG_COLOR = "#f0f0f0"
FG_COLOR = "#333333"
BTN_COLOR = "#e0e0e0"
FONT = "Times New Roman"

class TranslationApp:
    def __init__(self, ru2en_model, en2ru_model, ru_vocab, en_vocab):
        self.ru2en_model = ru2en_model
        self.en2ru_model = en2ru_model
        self.ru_vocab = ru_vocab
        self.en_vocab = en_vocab

        self.direction = "ru2en" # направление перевода (по умолчанию с русского на английский)

        self.build()

    def build(self):
        self.root = tk.Tk() # создание главного окна
        self.root.title("Переводчик RU ⇄ EN") # установка заголовка окна
        self.root.geometry("900x500") # установка размера окна
        self.root.configure(bg=BG_COLOR) # установка цвета фона (светло - серый)

        # Заголовок
        self.title_label = tk.Label(
            self.root, text="Переводчик RU ⇄ EN",
            font=(FONT, 16, "bold", "underline"), bg=BG_COLOR, # светло-серый цвет фона
            fg=FG_COLOR, # белый цвет текста
        )
        self.title_label.pack(pady=10) # отступ сверху

        # Создаем контейнер для левой и правой части
        main_frame = tk.Frame(self.root, bg=BG_COLOR)
        main_frame.pack(padx=10, fill=tk.BOTH, expand=True) # expand=True позволяет контейнеру растягиваться

        # Верх: кнопка swap по центру
        top_frame = tk.Frame(main_frame, bg=BG_COLOR)
        top_frame.pack(side=tk.TOP, fill=tk.X)   # ← сверху, на всю ширину

        swap_btn = tk.Button(
            top_frame, text="⇄",
            font=(FONT, 12, "bold"),
            command=self.swap_languages,
            bg=BTN_COLOR,
            fg=FG_COLOR,
            padx=15, pady=5,
            relief=tk.FLAT,                             # ← плоский стиль (опционально)
            activebackground="#d0d0d0"                  # ← цвет при нажатии
        )
        swap_btn.pack(anchor=tk.N, pady=5) # anchor=tk.N — прижимаем к верхнему краю

        # Создаем контейнер для текстовых полей
        middle_frame = tk.Frame(main_frame, bg=BG_COLOR)
        middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Создаем контейнер для левой части
        left_frame = tk.Frame(middle_frame, bg=BG_COLOR)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Создаем подпись и поле ввода в левой части
        self.left_label = tk.Label(
            left_frame, text="Русский", font=(FONT, 12), bg=BG_COLOR, fg=FG_COLOR
        )
        self.left_label.pack(side=tk.TOP, anchor=tk.W) #  прижимаем к верхнему краю

        text_frame_left = tk.Frame(left_frame, bg=BG_COLOR)
        text_frame_left.pack(fill=tk.BOTH, expand=True)

        # Создаем скроллбар справа по вертикали
        scrollbar_left = tk.Scrollbar(text_frame_left, orient=tk.VERTICAL)
        scrollbar_left.pack(side=tk.RIGHT, fill=tk.Y) # fill=tk.Y - растягивает по вертикали по всему фрейму

        # создаем текстовое поле с скроллбаром
        self.input_field = tk.Text(
            text_frame_left, height=12, width=40, # 12 строк, 40 символов
            bg="white", fg=FG_COLOR,
            font=(FONT, 13), wrap=tk.WORD, # перенос по словам
            yscrollcommand=scrollbar_left.set # привязываем скроллбар к текстовому полю
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_left.config(command=self.input_field.yview) # привязываем скроллбар к текстовому полю

        # Создаем контейнер для правой части
        right_frame = tk.Frame(middle_frame, bg=BG_COLOR)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10,0))

        # Создаем подпись и поле ввода в правой части
        self.right_label = tk.Label(
            right_frame, text="Английский", font=(FONT, 12), bg=BG_COLOR, fg=FG_COLOR
        )
        self.right_label.pack(side=tk.TOP, anchor=tk.W) #  прижимаем к левому краю

        text_frame_right = tk.Frame(right_frame, bg=BG_COLOR)
        text_frame_right.pack(fill=tk.BOTH, expand=True)

        # Создаем скроллбар справа по вертикали
        scrollbar_right = tk.Scrollbar(text_frame_right, orient=tk.VERTICAL)
        scrollbar_right.pack(side=tk.RIGHT, fill=tk.Y) # fill=tk.Y - растягивает по вертикали по всему фрейму

        # создаем текстовое поле с скроллбаром
        self.output_field = tk.Text(
            text_frame_right, height=12, width=40, # 12 строк, 40 символов
            bg="white", fg=FG_COLOR,
            font=(FONT, 13), wrap=tk.WORD, # перенос по словам
            yscrollcommand=scrollbar_right.set # привязываем скроллбар к текстовому полю
        )
        self.output_field.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_right.config(command=self.output_field.yview) # привязываем скроллбар к текстовому полю

        # Фрейм для кнопок
        button_frame = tk.Frame(main_frame, bg=BG_COLOR)
        button_frame.pack(side=tk.BOTTOM, anchor=tk.E, padx=20, pady=10)

        # кнопка "Перевести" справа
        translate_btn = tk.Button(
            button_frame, text="Перевести", font=(FONT, 12, "bold"),
            command=self.translate_text, bg=BTN_COLOR, fg=FG_COLOR,
            padx=15, pady=5,
            relief=tk.FLAT,                             # ← плоский стиль (опционально)
            activebackground="#d0d0d0"                  # ← цвет при нажатии
        )
        translate_btn.pack(side=tk.LEFT, padx=5)


        # кнопка "Очистить все" справа
        clear_btn = tk.Button(
            button_frame, text="Очистить все", font=(FONT, 12, "bold"),
            command=self.clear_all, bg=BTN_COLOR, fg=FG_COLOR,
            padx=15, pady=5,
            relief=tk.FLAT,                             # ← плоский стиль (опционально)
            activebackground="#d0d0d0"                  # ← цвет при нажатии
        )
        clear_btn.pack(side=tk.LEFT, padx=5)

    def translate_text(self):
        text = self.input_field.get("1.0", tk.END).strip() # типо получаем текс с  самого начала и до конца и обрезаем перенос строки в конце
        if not text:
            return

        try:
            if self.direction == "ru2en":
                result = self.ru2en_model.translate(
                    [text],
                    src_vocab=self.ru_vocab, tgt_vocab=self.en_vocab,
                    device=DEVICE, beam_size=BEAM_SIZE, max_len=MAX_LEN
                )
            else:
                result = self.en2ru_model.translate(
                    [text],
                    src_vocab=self.en_vocab, tgt_vocab=self.ru_vocab,
                    device=DEVICE, beam_size=BEAM_SIZE, max_len=MAX_LEN
                )

            self.output_field.delete("1.0", tk.END)
            self.output_field.insert("1.0", result[0])
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Ошибка", f"Не удалось перевести:\n{e}")

    def swap_languages(self):
        """Меняет направление перевода и меняет тексты полей местами."""
        self.direction = "en2ru" if self.direction == "ru2en" else "ru2en"

        left_text = self.input_field.get("1.0", tk.END).strip()
        right_text = self.output_field.get("1.0", tk.END).strip()

        self.input_field.delete("1.0", tk.END)
        self.input_field.insert("1.0", right_text)

        self.output_field.delete("1.0", tk.END)
        self.output_field.insert("1.0", left_text)

        if self.direction == "ru2en":
            self.left_label.config(text="Русский") # меняем текст над полями
            self.right_label.config(text="Английский")
        else:
            self.left_label.config(text="Английский")
            self.right_label.config(text="Русский")

    def clear_all(self):
        self.input_field.delete("1.0", tk.END)
        self.output_field.delete("1.0", tk.END)

    def run(self):
        '''Запускает главный (бесконечный) цикл приложения до закрытия окна'''
        self.root.mainloop()
