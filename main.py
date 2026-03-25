import os
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from faster_whisper import WhisperModel

# --- ПОРТАТИВНЫЙ FFMPEG ---
base_path = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + base_path # Добавляем папку проекта в PATH для FFmpeg

class TranscriberApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Voice to Text - Russian Edition")
        self.geometry("700x500")
        ctk.set_appearance_mode("dark")

        # Настройки модели
        self.model_size = "medium" # На 16ГБ VRAM можно даже "large-v3", но medium быстрее
        self.device = "cuda" # Используем твою 5060 Ti
        
        # Интерфейс
        self.grid_columnconfigure(0, weight=1)
        
        self.label = ctk.CTkLabel(self, text="Выберите аудиофайл (mp3, wav, ogg)", font=("Arial", 16))
        self.label.pack(pady=20)

        self.select_btn = ctk.CTkButton(self, text="Выбрать файл", command=self.select_file)
        self.select_btn.pack(pady=10)

        self.file_label = ctk.CTkLabel(self, text="Файл не выбран", font=("Arial", 12), text_color="gray")
        self.file_label.pack(pady=5)

        self.progress_bar = ctk.CTkProgressBar(self, width=400)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=20)

        self.textbox = ctk.CTkTextbox(self, width=600, height=200)
        self.textbox.pack(pady=10, padx=20)

        self.status_label = ctk.CTkLabel(self, text="Готов к работе")
        self.status_label.pack(pady=10)

        self.file_path = None

    def select_file(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.mp3 *.wav *.ogg *.m4a")]
        )
        if self.file_path:
            self.file_label.configure(text=os.path.basename(self.file_path))
            self.start_btn = ctk.CTkButton(self, text="Начать транскрибацию", fg_color="green", command=self.start_thread)
            self.start_btn.pack(pady=10)

    def start_thread(self):
        # Запускаем в отдельном потоке, чтобы GUI не зависал
        threading.Thread(target=self.transcribe, daemon=True).start()

    def transcribe(self):
        try:
            self.status_label.configure(text="Загрузка модели (это может занять время)...")
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start()
            
            # Инициализация модели
            model = WhisperModel(self.model_size, device=self.device, compute_type="float16")
            
            self.status_label.configure(text="Идет транскрибация...")
            
            segments, info = model.transcribe(self.file_path, beam_size=5, language="ru")
            
            full_text = ""
            for segment in segments:
                full_text += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
                # Обновляем текст в реальном времени
                self.textbox.delete("0.0", "end")
                self.textbox.insert("0.0", full_text)
            
            self.progress_bar.stop()
            self.progress_bar.set(1)
            self.status_label.configure(text="Готово!")
            
            # Сохранение в файл рядом с оригиналом
            out_file = self.file_path.rsplit('.', 1)[0] + ".txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            messagebox.showinfo("Успех", f"Текст сохранен в {out_file}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")
            self.status_label.configure(text="Ошибка")
            self.progress_bar.stop()

if __name__ == "__main__":
    app = TranscriberApp()
    app.mainloop()