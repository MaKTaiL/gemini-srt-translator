import gemini_srt_translator as gst
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import messagebox
import os
import webbrowser

class TranslatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Subtitle Translator Pro")
        self.root.geometry("600x500")
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure("Title.TLabel", font=("Helvetica", 14, "bold"))
        self.style.configure("Header.TLabel", font=("Helvetica", 11))
        self.style.configure("Status.TLabel", font=("Helvetica", 10))
        
        # Title
        title_frame = ttk.Frame(root)
        title_frame.pack(fill="x", padx=20, pady=(15,5))
        ttk.Label(title_frame, text="Subtitle Translator Pro", style="Title.TLabel").pack(side="left")
        
        # Main container
        main_container = ttk.Frame(root)
        main_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        # API Key section
        api_frame = ttk.LabelFrame(main_container, text="API Configuration", padding="10")
        api_frame.pack(fill="x", pady=(0,10))
        
        api_input_frame = ttk.Frame(api_frame)
        api_input_frame.pack(fill="x")
        
        ttk.Label(api_input_frame, text="Gemini API Key:", style="Header.TLabel").pack(side="left")
        self.api_key = tk.StringVar()
        self.api_entry = ttk.Entry(api_input_frame, textvariable=self.api_key, width=45, show="*")
        self.api_entry.pack(side="left", padx=10)
        
        api_btn_frame = ttk.Frame(api_input_frame)
        api_btn_frame.pack(side="left")
        
        # Show/Hide API Key
        self.show_api = tk.BooleanVar()
        ttk.Checkbutton(api_btn_frame, text="Show Key", 
                       variable=self.show_api, 
                       command=self.toggle_api_visibility).pack(side="left", padx=5)
        
        # Get API Key button
        ttk.Button(api_btn_frame, text="Get API Key", 
                  command=self.open_api_page).pack(side="left", padx=5)
        
        # File selection
        file_frame = ttk.LabelFrame(main_container, text="Files to Translate", padding="10")
        file_frame.pack(fill="both", expand=True, pady=(0,10))
        
        # File list with scrollbar
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill="both", expand=True, pady=(0,10))
        
        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, 
                                      font=("Helvetica", 10),
                                      background="white",
                                      selectbackground="#0078D7")
        self.file_listbox.pack(side="left", fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", 
                                command=self.file_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.file_listbox.config(yscrollcommand=scrollbar.set)
        
        # File buttons
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill="x")
        
        ttk.Button(btn_frame, text="Add Files", 
                  command=self.browse_files).pack(side="left", padx=(0,5))
        ttk.Button(btn_frame, text="Clear List", 
                  command=self.clear_files).pack(side="left")
        
        # Settings section
        settings_frame = ttk.LabelFrame(main_container, text="Translation Settings", padding="10")
        settings_frame.pack(fill="x", pady=(0,10))
        
        ttk.Label(settings_frame, text="Target Language:", 
                 style="Header.TLabel").pack(side="left")
        
        self.languages = ["English", "Chinese", "Japanese", "Korean", 
                         "French", "German", "Spanish", "Italian", "Russian"]
        self.target_lang = tk.StringVar(value="English")
        self.lang_combo = ttk.Combobox(settings_frame, 
                                     textvariable=self.target_lang,
                                     values=self.languages,
                                     state="readonly",
                                     width=30)
        self.lang_combo.pack(side="left", padx=10)
        
        # Progress section
        progress_frame = ttk.Frame(main_container)
        progress_frame.pack(fill="x", pady=(0,10))
        
        self.status_var = tk.StringVar(value="Ready to translate")
        ttk.Label(progress_frame, textvariable=self.status_var, 
                 style="Status.TLabel").pack(side="left")
        
        self.progress = ttk.Progressbar(progress_frame, mode='determinate', 
                                      length=200)
        self.progress.pack(side="right", padx=5)
        
        # Translate button
        self.translate_btn = ttk.Button(main_container, 
                                      text="Start Translation", 
                                      command=self.start_translation)
        self.translate_btn.pack(fill="x", pady=(0,10))
        
    def toggle_api_visibility(self):
        self.api_entry.config(show="" if self.show_api.get() else "*")
        
    def open_api_page(self):
        webbrowser.open("https://makersuite.google.com/app/apikey")
        messagebox.showinfo("Get API Key", 
                          "1. Sign in with your Google account\n"
                          "2. Create a new API key\n"
                          "3. Copy and paste it here")
        
    def browse_files(self):
        filenames = filedialog.askopenfilenames(
            title="Select Subtitle Files",
            filetypes=[("SRT files", "*.srt"), ("All files", "*.*")]
        )
        for filename in filenames:
            if filename not in self.file_listbox.get(0, tk.END):
                self.file_listbox.insert(tk.END, filename)
    
    def clear_files(self):
        self.file_listbox.delete(0, tk.END)
            
    def start_translation(self):
        try:
            files = list(self.file_listbox.get(0, tk.END))
            if not files:
                messagebox.showerror("Error", "Please select files to translate")
                return
            
            api_key = self.api_key.get().strip()
            if not api_key:
                messagebox.showerror("Error", "Please enter your API key")
                return
            
            # 语言代码映射
            language_codes = {
                "English": "en",
                "Chinese": "zh",
                "Japanese": "jp",
                "Korean": "kr",
                "French": "fr",
                "German": "de",
                "Spanish": "es",
                "Italian": "it",
                "Russian": "ru"
            }
            
            # 获取目标语言代码
            target_lang = self.target_lang.get()
            lang_code = language_codes.get(target_lang, "tr")
            
            # Set translation parameters
            gst.gemini_api_key = api_key
            gst.target_language = target_lang
            
            total_files = len(files)
            self.progress["maximum"] = total_files
            
            for index, file in enumerate(files, 1):
                # 生成新的输出文件名
                file_dir = os.path.dirname(file)
                file_name = os.path.basename(file)
                base_name, ext = os.path.splitext(file_name)
                output_file = os.path.join(file_dir, f"{base_name}.{lang_code}{ext}")
                
                self.status_var.set(f"Translating: {file_name} ({index}/{total_files})")
                self.progress["value"] = index
                self.root.update()
                
                # 设置输入输出文件
                gst.input_file = file
                gst.output_file = output_file
                
                # 开始翻译
                gst.translate()
            
            self.progress["value"] = 0
            self.status_var.set("Translation completed!")
            messagebox.showinfo("Success", f"Successfully translated {total_files} file(s)!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Translation error: {str(e)}")
            self.status_var.set("Translation failed")
            self.progress["value"] = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorGUI(root)
    root.mainloop()