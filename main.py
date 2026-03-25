import os
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
from faster_whisper import WhisperModel
import torch
import pynvml 
import pyperclip 

# --- ИНИЦИАЛИЗАЦИЯ NVIDIA GPU MONITOR ---
try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_NVML = True
except:
    HAS_NVML = False

base_path = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + base_path

class SpeechToTextApp(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        # Инициализируем оба родительских класса
        super().__init__()
        self.TkinterDnD_Init() # Исправленная строка инициализации DnD

        self.title("Speech2Text_localAI - RTX 5060 Ti Edition")
        self.geometry("850x700")
        ctk.set_appearance_mode("dark")

        self.model_size = "medium"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.grid_columnconfigure(0, weight=1)
        
        self.title_label = ctk.CTkLabel(self, text="AI Local Transcription", font=("Arial Bold", 22))
        self.title_label.pack(pady=(20, 5))

        self.action_btn = ctk.CTkButton(
            self, text="Выбрать файл или перетащить его сюда", 
            font=("Arial Bold", 14), height=60, command=self.browse_file
        )
        self.action_btn.pack(pady=10, padx=40, fill="x")
        
        # Регистрация Drag-and-Drop
        self.action_btn.drop_target_register(DND_FILES)
        self.action_btn.dnd_bind('<<Drop>>', self.handle_drop)

        self.textbox = ctk.CTkTextbox(self, width=750, height=380, font=("Arial", 14))
        self.textbox.pack(pady=10, padx=20, fill="both", expand=True)

        self.tool_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.tool_frame.pack(fill="x", padx=20, pady=5)

        self.copy_btn = ctk.CTkButton(self.tool_frame, text="Копировать текст", width=150, command=self.copy_to_clipboard)
        self.copy_btn.pack(side="left")

        self.vram_label = ctk.CTkLabel(self.tool_frame, text="VRAM: ---", text_color="#00FF00")
        self.vram_label.pack(side="right")

        self.progress_bar = ctk.CTkProgressBar(self, width=700)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=15)

        self.status_label = ctk.CTkLabel(self, text=f"Готов (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
        self.status_label.pack(pady=(0, 15))

        if HAS_NVML:
            self.update_vram_info()

    def update_vram_info(self):
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.vram_label.configure(text=f"VRAM: {info.used / 1024**3:.2f} / {info.total / 1024**3:.0f} GB")
        except: pass
        self.after(2000, self.update_vram_info)

    def copy_to_clipboard(self):
        text = self.textbox.get("0.0", "end").strip()
        if text:
            pyperclip.copy(text)
            self.status_label.configure(text="Текст в буфере!")
            self.after(2000, lambda: self.status_label.configure(text="Готов"))

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav *.ogg *.m4a")])
        if path: self.start_processing(path)

    def handle_drop(self, event):
        path = event.data.strip('{}') # Очистка пути Windows
        if os.path.isfile(path): self.start_processing(path)

    def start_processing(self, file_path):
        self.action_btn.configure(state="disabled", text="Идет обработка...")
        self.textbox.delete("0.0", "end")
        threading.Thread(target=self.transcribe, args=(file_path,), daemon=True).start()

    def transcribe(self, file_path):
        try:
            model = WhisperModel(self.model_size, device=self.device, compute_type="float16")
            segments, _ = model.transcribe(file_path, beam_size=5, language="ru")
            
            full_text = ""
            for segment in segments:
                full_text += f"{segment.text.strip()} "
                self.textbox.delete("0.0", "end")
                self.textbox.insert("0.0", full_text)
                self.textbox.see("end")
            
            self.status_label.configure(text="Завершено!")
            self.action_btn.configure(state="normal", text="Выбрать файл или перетащить его сюда")
            
            with open(file_path.rsplit('.', 1)[0] + "_text.txt", "w", encoding="utf-8") as f:
                f.write(full_text)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.action_btn.configure(state="normal")

if __name__ == "__main__":
    app = SpeechToTextApp()
    app.mainloop()