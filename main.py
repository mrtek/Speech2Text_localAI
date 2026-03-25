import os
import sys
import threading
import subprocess
import customtkinter as ctk
from tkinter import filedialog, messagebox
import windnd
from faster_whisper import WhisperModel
import torch
import pynvml 
import psutil 
import pyperclip 

# --- ИНИЦИАЛИЗАЦИЯ МОНИТОРИНГА ---
try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_NVML = True
except Exception:
    HAS_NVML = False

base_path = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + base_path

class UltimateSpeechApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Speech2Text_localAI - Ultimate Dashboard")
        self.geometry("1000x900")
        ctk.set_appearance_mode("dark")
        
        self.selected_file_path = None
        self.is_processing = False
        
        # Данные железа для авто-настройки
        self.vram_total = 0
        if HAS_NVML:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.vram_total = info.total / 1024**3
            except: pass

        self.grid_columnconfigure(0, weight=1)
        
        # --- 1. ПАНЕЛЬ НАСТРОЕК ---
        self.settings_frame = ctk.CTkFrame(self)
        self.settings_frame.pack(pady=10, padx=20, fill="x")

        # Модель
        ctk.CTkLabel(self.settings_frame, text="Модель:").grid(row=0, column=0, padx=10, pady=10)
        self.model_var = ctk.StringVar()
        self.model_menu = ctk.CTkOptionMenu(self.settings_frame, variable=self.model_var, 
                                            values=["tiny", "base", "small", "medium", "large-v3"])
        self.model_menu.grid(row=0, column=1, padx=5, pady=10)

        # Язык
        ctk.CTkLabel(self.settings_frame, text="Язык:").grid(row=0, column=2, padx=10, pady=10)
        self.lang_var = ctk.StringVar(value="ru")
        self.lang_menu = ctk.CTkOptionMenu(self.settings_frame, variable=self.lang_var, 
                                           values=["ru", "en", "de", "fr", "es", "it", "auto"])
        self.lang_menu.grid(row=0, column=3, padx=5, pady=10)

        # Устройство
        ctk.CTkLabel(self.settings_frame, text="Девайс:").grid(row=1, column=0, padx=10, pady=5)
        self.device_var = ctk.StringVar()
        self.device_menu = ctk.CTkOptionMenu(self.settings_frame, variable=self.device_var, values=["cuda", "cpu"])
        self.device_menu.grid(row=1, column=1, padx=5, pady=5)

        # Точность
        ctk.CTkLabel(self.settings_frame, text="Точность:").grid(row=1, column=2, padx=10, pady=5)
        self.compute_var = ctk.StringVar()
        self.compute_menu = ctk.CTkOptionMenu(self.settings_frame, variable=self.compute_var, 
                                               values=["float16", "int8", "float32"])
        self.compute_menu.grid(row=1, column=3, padx=5, pady=5)

        # Кнопка Авто-настройки
        self.auto_cfg_btn = ctk.CTkButton(self.settings_frame, text="💡 Рекомендовать", 
                                          fg_color="#1f538d", command=self.apply_best_settings)
        self.auto_cfg_btn.grid(row=0, column=4, rowspan=2, padx=20, pady=10)

        # --- 2. УПРАВЛЕНИЕ ---
        self.select_btn = ctk.CTkButton(self, text="ВЫБРАТЬ ФАЙЛ ИЛИ ПЕРЕТАЩИТЬ СЮДА", 
                                        font=("Arial Bold", 14), height=60, command=self.browse_file)
        self.select_btn.pack(pady=10, padx=20, fill="x")
        windnd.hook_dropfiles(self, self.handle_drop)

        self.file_info_label = ctk.CTkLabel(self, text="Файл не выбран", font=("Arial", 13), text_color="gray")
        self.file_info_label.pack()

        self.start_btn = ctk.CTkButton(self, text="НАЧАТЬ ТРАНСКРИБАЦИЮ", fg_color="#28a745", 
                                        font=("Arial Bold", 18), state="disabled", height=55, command=self.start_processing)
        self.start_btn.pack(pady=10, padx=100, fill="x")

        # Поле текста
        self.textbox = ctk.CTkTextbox(self, font=("Arial", 14), spacing2=5)
        self.textbox.pack(pady=10, padx=20, fill="both", expand=True)

        # Кнопки под текстом
        self.control_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.control_frame.pack(fill="x", padx=20, pady=5)

        ctk.CTkButton(self.control_frame, text="Копировать текст", command=self.copy_text).pack(side="left", padx=5)
        ctk.CTkButton(self.control_frame, text="Сбросить всё", fg_color="#dc3545", 
                      hover_color="#c82333", command=self.reset_app).pack(side="left", padx=5)

        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10, padx=20, fill="x")

        # --- 3. DASHBOARD МОНИТОРИНГА ---
        self.monitor_frame = ctk.CTkFrame(self, height=80)
        self.monitor_frame.pack(pady=(5, 15), padx=20, fill="x")
        for i in range(4): self.monitor_frame.grid_columnconfigure(i, weight=1)

        self.cpu_label = ctk.CTkLabel(self.monitor_frame, text="CPU: --%", font=("Consolas", 12))
        self.cpu_label.grid(row=0, column=0, pady=10)

        self.ram_label = ctk.CTkLabel(self.monitor_frame, text="RAM: -- GB", font=("Consolas", 12))
        self.ram_label.grid(row=0, column=1, pady=10)

        self.gpu_label = ctk.CTkLabel(self.monitor_frame, text="GPU: --% | --°C", font=("Consolas", 12))
        self.gpu_label.grid(row=0, column=2, pady=10)

        self.vram_label = ctk.CTkLabel(self.monitor_frame, text="VRAM: -- GB", font=("Consolas", 12), text_color="#00FF00")
        self.vram_label.grid(row=0, column=3, pady=10)

        self.status_label = ctk.CTkLabel(self, text="Ожидание...", font=("Arial", 12))
        self.status_label.pack(pady=(0, 10))

        # Инициализация
        self.apply_best_settings()
        self.update_metrics()

    # --- ИНТЕЛЛЕКТУАЛЬНАЯ НАСТРОЙКА ---
    def apply_best_settings(self):
        """Автоматический подбор настроек под железо"""
        if torch.cuda.is_available():
            self.device_var.set("cuda")
            if self.vram_total >= 10: # Для твоей 5060 Ti 16GB
                self.model_var.set("large-v3")
                self.compute_var.set("float16")
            elif self.vram_total >= 6:
                self.model_var.set("medium")
                self.compute_var.set("float16")
            else:
                self.model_var.set("small")
                self.compute_var.set("int8")
        else:
            self.device_var.set("cpu")
            self.model_var.set("small")
            self.compute_var.set("int8")
        self.status_label.configure(text="Настройки оптимизированы под ваше железо")

    # --- МОНИТОРИН Г ---
    def update_metrics(self):
        cpu_load = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        self.cpu_label.configure(text=f"CPU: {cpu_load}%")
        self.ram_label.configure(text=f"RAM: {ram.used/1024**3:.1f}/{ram.total/1024**3:.1f} GB")

        if HAS_NVML:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.gpu_label.configure(text=f"GPU: {util.gpu}% | {temp}°C")
                self.vram_label.configure(text=f"VRAM: {mem.used/1024**3:.2f}/{mem.total/1024**3:.1f} GB")
            except: pass
        self.after(2000, self.update_metrics)

    # --- ЛОГИКА ---
    def reset_app(self):
        self.selected_file_path = None
        self.textbox.delete("0.0", "end")
        self.file_info_label.configure(text="Файл не выбран", text_color="gray")
        self.start_btn.configure(state="disabled")
        self.progress_bar.set(0)
        self.status_label.configure(text="Готово к новой задаче")

    def handle_drop(self, files):
        if self.is_processing: return
        try: path = files[0].decode('cp1251')
        except: path = files[0].decode('utf-8', errors='ignore')
        self.after(100, lambda: self.set_selected_file(path))

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Media", "*.mp3 *.wav *.ogg *.m4a *.flac *.mp4 *.mkv")])
        if path: self.set_selected_file(path)

    def set_selected_file(self, path):
        self.selected_file_path = path
        self.file_info_label.configure(text=f"ВЫБРАН: {os.path.basename(path)}", text_color="#50FA7B")
        self.start_btn.configure(state="normal")

    def start_processing(self):
        if self.is_processing: return
        dev, comp = self.device_var.get(), self.compute_var.get()
        if dev == "cpu" and comp == "float16": 
            self.compute_var.set("int8")
            comp = "int8"
            
        self.is_processing = True
        self.start_btn.configure(state="disabled", text="В ПРОЦЕССЕ...")
        self.textbox.delete("0.0", "end")
        threading.Thread(target=self.transcribe_worker, args=(dev, comp), daemon=True).start()

    def transcribe_worker(self, dev, comp):
        try:
            self.after(0, self.progress_bar.start)
            m_size, lang = self.model_var.get(), self.lang_var.get()
            
            self.after(0, lambda: self.status_label.configure(text=f"Инициализация {m_size}..."))
            model = WhisperModel(m_size, device=dev, compute_type=comp)
            
            self.after(0, lambda: self.status_label.configure(text="Идет расшифровка..."))
            segments, _ = model.transcribe(self.selected_file_path, beam_size=5, 
                                           language=None if lang == "auto" else lang)
            
            full_text = ""
            for segment in segments:
                full_text += f"{segment.text.strip()} "
                self.after(0, lambda t=full_text: self.update_ui_text(t))
            
            output_file = self.selected_file_path.rsplit('.', 1)[0] + "_text.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            self.after(0, lambda: self.status_label.configure(text="Завершено! Открываю файл..."))
            # Авто-открытие в Блокноте
            os.startfile(output_file)

        except Exception as e:
            msg = str(e)
            self.after(0, lambda m=msg: messagebox.showerror("Ошибка", f"Критический сбой: {m}"))
        finally:
            self.is_processing = False
            self.after(0, self.progress_bar.stop)
            self.after(0, lambda: self.progress_bar.set(1))
            self.after(0, lambda: self.start_btn.configure(state="normal", text="НАЧАТЬ ТРАНСКРИБАЦИЮ"))

    def update_ui_text(self, text):
        self.textbox.delete("0.0", "end")
        self.textbox.insert("0.0", text)
        self.textbox.see("end")

    def copy_text(self):
        txt = self.textbox.get("0.0", "end").strip()
        if txt:
            pyperclip.copy(txt)
            self.status_label.configure(text="Скопировано!")

if __name__ == "__main__":
    app = UltimateSpeechApp()
    app.mainloop()