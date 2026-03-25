import os
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES # Добавили инструменты для DnD
from faster_whisper import WhisperModel
import torch

# --- ПОРТАТИВНЫЙ FFMPEG ---
base_path = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + base_path

# Используем специальный класс для поддержки Drag-and-Drop
class SpeechToTextApp(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._pyTkinterDnD.TkinterDnD_Init(self)

        # Настройки окна
        self.title("Speech2Text_localAI")
        self.geometry("800x650")
        ctk.set_appearance_mode("dark")

        # Инициализация ИИ
        self.model_size = "medium"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # UI
        self.grid_columnconfigure(0, weight=1)
        
        self.title_label = ctk.CTkLabel(self, text="AI Transcription (Drag & Drop)", font=("Arial Bold", 20))
        self.title_label.pack(pady=(20, 10))

        # Кнопка (она же зона сброса)
        self.action_btn = ctk.CTkButton(
            self, 
            text="Выбрать файл или перетащить его сюда", 
            font=("Arial Bold", 14),
            height=60, # Сделали чуть выше для удобства попадания
            command=self.browse_file
        )
        self.action_btn.pack(pady=10, padx=40, fill="x")

        # Регистрируем кнопку как цель для перетаскивания файлов
        self.action_btn.drop_target_register(DND_FILES)
        self.action_btn.dnd_bind('<<Drop>>', self.handle_drop)

        self.file_label = ctk.CTkLabel(self, text="Ожидание файла...", font=("Arial", 12), text_color="gray")
        self.file_label.pack(pady=5)

        self.progress_bar = ctk.CTkProgressBar(self, width=600)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=20)

        self.textbox = ctk.CTkTextbox(self, width=700, height=350, font=("Arial", 14))
        self.textbox.pack(pady=10, padx=20, fill="both", expand=True)

        self.status_label = ctk.CTkLabel(self, text="Готов")
        self.status_label.pack(pady=(5, 20))

    def browse_file(self):
        """Обычный выбор через диалоговое окно"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Аудио", "*.mp3 *.wav *.ogg *.m4a *.flac")]
        )
        if file_path:
            self.start_processing(file_path)

    def handle_drop(self, event):
        """Логика обработки перетащенного файла"""
        file_path = event.data
        
        # Windows иногда добавляет фигурные скобки {}, если в пути есть пробелы
        if file_path.startswith('{') and file_path.endswith('}'):
            file_path = file_path[1:-1]
            
        # Проверка расширения
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.mp3', '.wav', '.ogg', '.m4a', '.flac']:
            self.start_processing(file_path)
        else:
            messagebox.showwarning("Формат", "Пожалуйста, перетащите аудиофайл (mp3, wav, ogg и т.д.)")

    def start_processing(self, file_path):
        """Запуск процесса транскрибации"""
        self.file_label.configure(text=f"Файл: {os.path.basename(file_path)}")
        self.action_btn.configure(state="disabled", text="Обработка...")
        self.textbox.delete("0.0", "end")
        threading.Thread(target=self.transcribe, args=(file_path,), daemon=True).start()

    def transcribe(self, file_path):
        try:
            self.status_label.configure(text="Загрузка модели...")
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start()
            
            # На твоей 5060 Ti medium модель работает отлично
            model = WhisperModel(self.model_size, device=self.device, compute_type="float16")
            
            self.status_label.configure(text="Идет транскрибация...")
            segments, info = model.transcribe(file_path, beam_size=5, language="ru")
            
            full_text = ""
            for segment in segments:
                full_text += f"{segment.text.strip()} "
                # Обновление UI
                self.textbox.delete("0.0", "end")
                self.textbox.insert("0.0", full_text)
                self.textbox.see("end")
            
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self.progress_bar.set(1)
            self.status_label.configure(text="Завершено!")
            self.action_btn.configure(state="normal", text="Выбрать файл или перетащить его сюда")
            
            # Сохранение в файл
            out_file = file_path.rsplit('.', 1)[0] + "_text.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            messagebox.showinfo("Готово", "Текст успешно сохранен рядом с аудиофайлом.")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            self.status_label.configure(text="Ошибка")
            self.action_btn.configure(state="normal", text="Попробовать снова")
            self.progress_bar.stop()

if __name__ == "__main__":
    app = SpeechToTextApp()
    app.mainloop()