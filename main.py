import os
import sys
import threading
import subprocess
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image  # Для работы с png-кнопками
import windnd
from faster_whisper import WhisperModel
import torch
import pynvml 
import psutil 
import pyperclip 

# --- ИНИЦИАЛИЗАЦИЯ ПУТЕЙ ---
base_path = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(base_path, "models")
ICON_PATH = os.path.join(base_path, "icon.ico")
DARK_IMG_PATH = os.path.join(base_path, "dvs.png")
LIGHT_IMG_PATH = os.path.join(base_path, "lss.png")

os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["PATH"] += os.pathsep + base_path

# --- ИНИЦИАЛИЗАЦИЯ МОНИТОРИНГА ---
try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_NVML = True
except Exception:
    HAS_NVML = False

def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")

# --- ОКНО ЗАГРУЗКИ МОДЕЛЕЙ ---
class DownloadWindow(ctk.CTkToplevel):
    def __init__(self, parent, vram_total):
        super().__init__(parent)
        self.title("Менеджер моделей")
        width, height = 650, 600
        center_window(self, width, height)
        
        # Установка иконки для окна менеджера
        if os.path.exists(ICON_PATH):
            self.after(200, lambda: self.iconbitmap(ICON_PATH))
        
        self.after(100, self.lift)
        self.after(100, self.focus_force)
        self.grab_set()
        
        self.vram_total = vram_total
        self.models = ["tiny", "base", "small", "medium", "large-v3"]
        self.btns = {}
        self.status_labels = {}
        
        ctk.CTkLabel(self, text="Загрузка моделей Whisper", font=("Arial Bold", 22)).pack(pady=20)
        
        self.scroll_frame = ctk.CTkScrollableFrame(self, width=600, height=350)
        self.scroll_frame.pack(pady=10, padx=20, fill="both", expand=True)

        for m in self.models:
            self.render_model_row(m)

        self.overall_progress = ctk.CTkProgressBar(self, width=500)
        self.overall_progress.set(0)
        self.overall_progress.pack(pady=20)
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=20)
        
        ctk.CTkButton(btn_frame, text="Скачать рекомендованные", fg_color="#1f538d", 
                      command=self.download_recommended).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="Закрыть", command=self.destroy).pack(side="left", padx=10)

    def is_model_downloaded(self, model_name):
        target_path = os.path.join(MODELS_DIR, f"models--Systran--faster-whisper-{model_name}")
        return os.path.exists(target_path)

    def render_model_row(self, m):
        frame = ctk.CTkFrame(self.scroll_frame)
        frame.pack(pady=5, fill="x")
        is_rec = (m == "large-v3" and self.vram_total >= 10) or (m == "medium" and self.vram_total >= 6)
        label_text = f"{m.upper()}" + (" ⭐" if is_rec else "")
        lbl = ctk.CTkLabel(frame, text=label_text, width=120, anchor="w", font=("Arial Bold", 13))
        lbl.pack(side="left", padx=15, pady=10)

        exists = self.is_model_downloaded(m)
        status_text = "Готова" if exists else "Не скачана"
        status_color = "#28a745" if exists else "#aaaaaa"
        stat_lbl = ctk.CTkLabel(frame, text=status_text, text_color=status_color, width=100)
        stat_lbl.pack(side="left", padx=10)
        self.status_labels[m] = stat_lbl

        btn = ctk.CTkButton(frame, text="Скачать", width=100, command=lambda name=m: self.start_download(name))
        if exists: btn.configure(state="disabled", text="Есть")
        btn.pack(side="right", padx=15)
        self.btns[m] = btn

    def start_download(self, model_name):
        self.btns[model_name].configure(state="disabled", text="Загрузка...")
        self.status_labels[model_name].configure(text="Качаем...", text_color="#ffc107")
        self.overall_progress.configure(mode="indeterminate"); self.overall_progress.start()
        threading.Thread(target=self._download_task, args=(model_name,), daemon=True).start()

    def _download_task(self, model_name):
        try:
            WhisperModel(model_name, device="cpu", compute_type="int8", download_root=MODELS_DIR)
            self.after(0, lambda: self.finish_download(model_name, True))
        except Exception as e:
            self.after(0, lambda: self.finish_download(model_name, False, str(e)))

    def finish_download(self, model_name, success, err=""):
        self.overall_progress.stop(); self.overall_progress.configure(mode="determinate"); self.overall_progress.set(1 if success else 0)
        if success:
            self.status_labels[model_name].configure(text="Готова", text_color="#28a745")
            self.btns[model_name].configure(text="Есть", fg_color="#28a745")
        else:
            self.status_labels[model_name].configure(text="Ошибка", text_color="#dc3545")
            messagebox.showerror("Ошибка", f"Сбой загрузки: {err}")

    def download_recommended(self):
        m = "large-v3" if self.vram_total >= 10 else ("medium" if self.vram_total >= 6 else "small")
        if not self.is_model_downloaded(m): self.start_download(m)
        else: messagebox.showinfo("Инфо", "Модель уже в наличии.")

# --- ГЛАВНОЕ ОКНО ---
class UltimateSpeechApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Speech2Text_localAI - Ultimate Edition")
        width, height = 1000, 920
        center_window(self, width, height)
        
        # Установка иконки основного окна
        if os.path.exists(ICON_PATH):
            self.iconbitmap(ICON_PATH)
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.is_processing = False
        self.vram_total = 0
        if HAS_NVML:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.vram_total = info.total / 1024**3
            except: pass

        self.grid_columnconfigure(0, weight=1)
        
        # --- ПАНЕЛЬ НАСТРОЕК ---
        self.settings_frame = ctk.CTkFrame(self)
        self.settings_frame.pack(pady=15, padx=20, fill="x")
        self.settings_frame.grid_columnconfigure(4, weight=1) 
        self.settings_frame.grid_columnconfigure(5, weight=0) 

        btn_color = "#1f538d"

        # Колонки 0-1: Модель / Устройство
        ctk.CTkLabel(self.settings_frame, text="Модель:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.model_var = ctk.StringVar()
        ctk.CTkOptionMenu(self.settings_frame, variable=self.model_var, values=["tiny", "base", "small", "medium", "large-v3"]).grid(row=0, column=1, padx=5, sticky="w")

        ctk.CTkLabel(self.settings_frame, text="Устройство:").grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.device_var = ctk.StringVar()
        ctk.CTkOptionMenu(self.settings_frame, variable=self.device_var, values=["cuda", "cpu"]).grid(row=1, column=1, padx=5, sticky="w")

        # Колонки 2-3: Язык / Точность
        ctk.CTkLabel(self.settings_frame, text="Язык:").grid(row=0, column=2, padx=10, pady=10, sticky="e")
        self.lang_var = ctk.StringVar(value="ru")
        ctk.CTkOptionMenu(self.settings_frame, variable=self.lang_var, values=["ru", "en", "de", "fr", "auto"]).grid(row=0, column=3, padx=5, sticky="w")

        ctk.CTkLabel(self.settings_frame, text="Точность:").grid(row=1, column=2, padx=10, pady=10, sticky="e")
        self.compute_var = ctk.StringVar()
        ctk.CTkOptionMenu(self.settings_frame, variable=self.compute_var, values=["float16", "int8", "float32"]).grid(row=1, column=3, padx=5, sticky="w")

        # Колонка 4: PNG ПЕРЕКЛЮЧАТЕЛИ ТЕМЫ (Размер изменен на 60x60)
        self.theme_img_frame = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        self.theme_img_frame.grid(row=0, column=4, rowspan=2, padx=10)

        try:
            light_img = ctk.CTkImage(light_image=Image.open(LIGHT_IMG_PATH), size=(60, 60))
            dark_img = ctk.CTkImage(light_image=Image.open(DARK_IMG_PATH), size=(60, 60))

            self.light_btn = ctk.CTkButton(self.theme_img_frame, image=light_img, text="", width=60, height=60, 
                                            fg_color="transparent", hover_color="#333333", command=lambda: ctk.set_appearance_mode("light"))
            self.light_btn.pack(side="left", padx=5)

            self.dark_btn = ctk.CTkButton(self.theme_img_frame, image=dark_img, text="", width=60, height=60, 
                                           fg_color="transparent", hover_color="#333333", command=lambda: ctk.set_appearance_mode("dark"))
            self.dark_btn.pack(side="left", padx=5)
        except Exception:
            ctk.CTkLabel(self.theme_img_frame, text="[PNG MISSING]").pack()

        # Колонка 5: КНОПКИ ДЕЙСТВИЯ
        self.auto_cfg_btn = ctk.CTkButton(self.settings_frame, text="Автоконфигурация", fg_color=btn_color, command=self.apply_best_settings)
        self.auto_cfg_btn.grid(row=0, column=5, padx=20, pady=10, sticky="e")

        self.dl_window_btn = ctk.CTkButton(self.settings_frame, text="Скачать модели", fg_color=btn_color, command=self.open_dl_window)
        self.dl_window_btn.grid(row=1, column=5, padx=20, pady=10, sticky="e")

        # --- ЦЕНТРАЛЬНАЯ ЧАСТЬ ---
        self.select_btn = ctk.CTkButton(self, text="ВЫБРАТЬ ФАЙЛ ИЛИ ПЕРЕТАЩИТЬ СЮДА (mp3, wav, ogg)", font=("Arial Bold", 15), height=65, command=self.browse_file)
        self.select_btn.pack(pady=10, padx=20, fill="x")
        windnd.hook_dropfiles(self, self.handle_drop)

        self.file_info_label = ctk.CTkLabel(self, text="Файл не выбран", font=("Arial", 13), text_color="gray")
        self.file_info_label.pack()

        self.start_btn = ctk.CTkButton(self, text="НАЧАТЬ ТРАНСКРИБАЦИЮ", fg_color="#28a745", hover_color="#218838",
                                        font=("Arial Bold", 18), state="disabled", height=60, command=self.start_processing)
        self.start_btn.pack(pady=10, padx=100, fill="x")

        self.textbox = ctk.CTkTextbox(self, font=("Arial", 14), spacing2=5)
        self.textbox.pack(pady=10, padx=20, fill="both", expand=True)

        self.control_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.control_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.control_frame, text="Копировать текст", width=160, command=self.copy_text).pack(side="left", padx=5)
        ctk.CTkButton(self.control_frame, text="Очистить", fg_color="#dc3545", width=140, command=self.reset_app).pack(side="left", padx=5)

        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10, padx=20, fill="x")

        # --- DASHBOARD ---
        self.monitor_frame = ctk.CTkFrame(self, height=80)
        self.monitor_frame.pack(pady=(5, 10), padx=20, fill="x")
        for i in range(4): self.monitor_frame.grid_columnconfigure(i, weight=1)

        self.cpu_label = ctk.CTkLabel(self.monitor_frame, text="CPU: --%", font=("Consolas", 12)); self.cpu_label.grid(row=0, column=0, pady=10)
        self.ram_label = ctk.CTkLabel(self.monitor_frame, text="RAM: -- GB", font=("Consolas", 12)); self.ram_label.grid(row=0, column=1, pady=10)
        self.gpu_label = ctk.CTkLabel(self.monitor_frame, text="GPU: --% | --°C", font=("Consolas", 12)); self.gpu_label.grid(row=0, column=2, pady=10)
        self.vram_label = ctk.CTkLabel(self.monitor_frame, text="VRAM: -- GB", font=("Consolas", 12), text_color="#00FF00"); self.vram_label.grid(row=0, column=3, pady=10)

        # --- FOOTER ---
        self.footer_label = ctk.CTkLabel(self, text="by Jyborg with Gemini 2026", font=("Arial", 11), text_color="gray")
        self.footer_label.pack(side="right", padx=20, pady=(0, 10))

        self.status_label = ctk.CTkLabel(self, text="Готов к работе", font=("Arial", 12))
        self.status_label.pack(side="left", padx=20, pady=(0, 10))

        self.apply_best_settings()
        self.update_metrics()

    # --- ЛОГИКА ---
    def apply_best_settings(self):
        if torch.cuda.is_available():
            self.device_var.set("cuda")
            if self.vram_total >= 10: self.model_var.set("large-v3"); self.compute_var.set("float16")
            elif self.vram_total >= 6: self.model_var.set("medium"); self.compute_var.set("float16")
            else: self.model_var.set("small"); self.compute_var.set("int8")
        else:
            self.device_var.set("cpu"); self.model_var.set("small"); self.compute_var.set("int8")

    def open_dl_window(self): DownloadWindow(self, self.vram_total)

    def update_metrics(self):
        cpu_load = psutil.cpu_percent(); ram = psutil.virtual_memory()
        self.cpu_label.configure(text=f"CPU: {cpu_load}%")
        self.ram_label.configure(text=f"RAM: {ram.used/1024**3:.1f}/{ram.total/1024**3:.1f} GB")
        if HAS_NVML:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle); temp = pynvml.nvmlDeviceGetTemperature(handle, 0); mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.gpu_label.configure(text=f"GPU: {util.gpu}% | {temp}°C")
                self.vram_label.configure(text=f"VRAM: {mem.used/1024**3:.2f}/{mem.total/1024**3:.1f} GB")
            except: pass
        self.after(2000, self.update_metrics)

    def reset_app(self):
        self.selected_file_path = None; self.textbox.delete("0.0", "end")
        self.file_info_label.configure(text="Файл не выбран", text_color="gray"); self.start_btn.configure(state="disabled")
        self.progress_bar.set(0)

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
        if dev == "cpu" and comp == "float16": comp = "int8"; self.compute_var.set("int8")
        self.is_processing = True; self.start_btn.configure(state="disabled", text="В ПРОЦЕССЕ...")
        threading.Thread(target=self.transcribe_worker, args=(dev, comp), daemon=True).start()

    def transcribe_worker(self, dev, comp):
        try:
            self.after(0, self.progress_bar.start)
            m_size, lang = self.model_var.get(), self.lang_var.get()
            self.after(0, lambda: self.status_label.configure(text=f"Загрузка {m_size}..."))
            model = WhisperModel(m_size, device=dev, compute_type=comp, download_root=MODELS_DIR)
            self.after(0, lambda: self.status_label.configure(text="Расшифровка..."))
            segments, _ = model.transcribe(self.selected_file_path, beam_size=5, language=None if lang == "auto" else lang)
            full_text = ""
            for segment in segments:
                full_text += f"{segment.text.strip()} "
                self.after(0, lambda t=full_text: self.update_ui_text(t))
            output_file = self.selected_file_path.rsplit('.', 1)[0] + "_text.txt"
            with open(output_file, "w", encoding="utf-8") as f: f.write(full_text)
            os.startfile(output_file)
        except Exception as e: self.after(0, lambda m=str(e): messagebox.showerror("Ошибка", m))
        finally: self.is_processing = False; self.after(0, self.progress_bar.stop); self.after(0, lambda: self.start_btn.configure(state="normal", text="НАЧАТЬ ТРАНСКРИБАЦИЮ"))

    def update_ui_text(self, text):
        self.textbox.delete("0.0", "end"); self.textbox.insert("0.0", text); self.textbox.see("end")

    def copy_text(self):
        txt = self.textbox.get("0.0", "end").strip()
        if txt: pyperclip.copy(txt); self.status_label.configure(text="Скопировано!")

if __name__ == "__main__":
    app = UltimateSpeechApp()
    app.mainloop()