import os
import sys
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
import windnd
from faster_whisper import WhisperModel
import torch
import pynvml 
import pyperclip 

# --- ГЛОБАЛЬНЫЕ НАСТРОЙКИ ---
try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_NVML = True
except Exception:
    HAS_NVML = False

# Добавляем путь к FFmpeg
base_path = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + base_path

class SpeechToTextApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Speech2Text_localAI - Stable Pro")
        self.geometry("850x750")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.selected_file_path = None
        self.is_processing = False
        
        # UI
        self.grid_columnconfigure(0, weight=1)
        
        self.title_label = ctk.CTkLabel(self, text="AI Local Transcription", font=("Arial Bold", 24))
        self.title_label.pack(pady=(20, 10))

        self.select_btn = ctk.CTkButton(
            self, text="ВЫБРАТЬ ФАЙЛ ИЛИ ПЕРЕТАЩИТЬ СЮДА", 
            font=("Arial Bold", 14), height=70, fg_color="#3B3B3B", hover_color="#4B4B4B",
            command=self.browse_file
        )
        self.select_btn.pack(pady=10, padx=40, fill="x")

        # Регистрация Drag-and-Drop
        windnd.hook_dropfiles(self, self.handle_drop)

        self.file_info_label = ctk.CTkLabel(self, text="Файл не выбран", font=("Arial", 13), text_color="#AAAAAA")
        self.file_info_label.pack(pady=5)

        self.start_btn = ctk.CTkButton(
            self, text="НАЧАТЬ ТРАНСКРИБАЦИЮ", 
            font=("Arial Bold", 16), height=60, fg_color="#28a745", hover_color="#218838",
            state="disabled", command=self.start_processing
        )
        self.start_btn.pack(pady=10, padx=100, fill="x")

        self.textbox = ctk.CTkTextbox(self, width=750, height=350, font=("Arial", 14), spacing2=5)
        self.textbox.pack(pady=10, padx=20, fill="both", expand=True)

        self.tool_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.tool_frame.pack(fill="x", padx=20, pady=5)

        self.copy_btn = ctk.CTkButton(self.tool_frame, text="Копировать текст", width=180, command=self.copy_to_clipboard)
        self.copy_btn.pack(side="left")

        self.vram_label = ctk.CTkLabel(self.tool_frame, text="VRAM: ---", text_color="#00FF00", font=("Consolas", 14))
        self.vram_label.pack(side="right")

        self.progress_bar = ctk.CTkProgressBar(self, width=700)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=15)

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        self.status_label = ctk.CTkLabel(self, text=f"Устройство: {gpu_name}", font=("Arial", 12))
        self.status_label.pack(pady=(0, 15))

        if HAS_NVML:
            self.update_vram_info()

    # --- СЛУЖЕБНЫЕ МЕТОДЫ (ПОТОКОБЕЗОПАСНЫЕ) ---

    def safe_update_status(self, text):
        self.after(0, lambda: self.status_label.configure(text=text))

    def safe_update_textbox(self, text):
        def update():
            self.textbox.delete("0.0", "end")
            self.textbox.insert("0.0", text)
            self.textbox.see("end")
        self.after(0, update)

    def update_vram_info(self):
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.vram_label.configure(text=f"VRAM: {info.used / 1024**3:.2f} / {info.total / 1024**3:.0f} GB")
        except: pass
        self.after(2000, self.update_vram_info)

    # --- ЛОГИКА ФАЙЛОВ ---

    def handle_drop(self, files):
        if self.is_processing: return
        try:
            file_path = files[0].decode('cp1251')
        except:
            file_path = files[0].decode('utf-8', errors='ignore')
        # Используем after, чтобы выйти из контекста обратного вызова windnd
        self.after(100, lambda: self.set_selected_file(file_path))

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav *.ogg *.m4a *.flac")])
        if path: self.set_selected_file(path)

    def set_selected_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.mp3', '.wav', '.ogg', '.m4a', '.flac']:
            self.selected_file_path = path
            self.file_info_label.configure(text=f"ВЫБРАН: {os.path.basename(path)}", text_color="#50FA7B")
            self.start_btn.configure(state="normal")
            self.status_label.configure(text="Готов к запуску")
        else:
            messagebox.showwarning("Формат", "Файл не поддерживается.")

    def start_processing(self):
        if not self.selected_file_path or self.is_processing: return
        self.is_processing = True
        self.start_btn.configure(state="disabled", text="В ПРОЦЕССЕ...")
        self.select_btn.configure(state="disabled")
        self.textbox.delete("0.0", "end")
        
        threading.Thread(target=self.transcribe_worker, args=(self.selected_file_path,), daemon=True).start()

    # --- РАБОЧИЙ ПОТОК ИИ ---

    def transcribe_worker(self, file_path):
        try:
            self.after(0, self.progress_bar.start)
            self.safe_update_status("Загрузка ИИ...")
            
            # Модель создается прямо в потоке для стабильности GIL
            model = WhisperModel("medium", device="cuda", compute_type="float16")
            
            self.safe_update_status("Распознавание...")
            segments, _ = model.transcribe(file_path, beam_size=5, language="ru")
            
            full_text = ""
            for segment in segments:
                full_text += f"{segment.text.strip()} "
                self.safe_update_textbox(full_text)
            
            # Сохранение
            output_file = file_path.rsplit('.', 1)[0] + "_text.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            self.safe_update_status("Завершено! Текст сохранен.")

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
        finally:
            self.is_processing = False
            self.after(0, self.progress_bar.stop)
            self.after(0, lambda: self.progress_bar.set(1))
            self.after(0, lambda: self.start_btn.configure(state="normal", text="НАЧАТЬ ТРАНСКРИБАЦИЮ"))
            self.after(0, lambda: self.select_btn.configure(state="normal"))

    def copy_to_clipboard(self):
        text = self.textbox.get("0.0", "end").strip()
        if text:
            pyperclip.copy(text)
            self.status_label.configure(text="Скопировано!")

if __name__ == "__main__":
    app = SpeechToTextApp()
    app.mainloop()