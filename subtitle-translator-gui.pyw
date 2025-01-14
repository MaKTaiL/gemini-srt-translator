import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QTextEdit, QLabel, 
                           QFileDialog, QListWidget, QMessageBox, QLineEdit,
                           QComboBox, QProgressBar, QGroupBox, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QEvent
from PyQt6.QtGui import QPalette
import gemini_srt_translator as gst
from itertools import cycle
from datetime import datetime, timedelta
import json
class TranslationWorker(QThread):
    finished = pyqtSignal(str, int, float)
    progress = pyqtSignal(str, int, int)
    error = pyqtSignal(str, str)
    batch_progress = pyqtSignal(int, int)
    def __init__(self, input_file, api_key, target_language, batch_size=28, 
                 request_delay=0, model_name="gemini-1.5-flash", description=None):
        super().__init__()
        self.input_file = input_file
        self.api_key = api_key
        self.target_language = target_language
        self.start_time = None
        self.subtitle_count = 0
        self.request_delay = request_delay
        self.batch_size = batch_size
        self.model_name = model_name
        self.description = description
    def run(self):
        try:
            self.start_time = datetime.now()
            print(f"Counting subtitles in file: {self.input_file}")
            self.subtitle_count = self._count_subtitles()
            print(f"Found {self.subtitle_count} subtitles")
            
            print(f"Setting up translator with parameters:")
            print(f"- API Key: {self.api_key[:10]}...")
            print(f"- Target Language: {self.target_language}")
            print(f"- Model Name: {self.model_name}")
            translator = self._setup_translator()
            
            print("Starting translation...")
            translator.translate()
            
            duration = (datetime.now() - self.start_time).total_seconds()
            self.finished.emit(self.input_file, self.subtitle_count, duration)
            
        except Exception as e:
            error_type = type(e).__name__
            error_details = f"{error_type}: {str(e)}"
            full_error = (
                f"Error in translating {os.path.basename(self.input_file)}\n"
                f"Error Type: {error_type}\n"
                f"Error Message: {str(e)}\n"
                f"File Path: {self.input_file}\n"
                f"Target Language: {self.target_language}\n"
                f"Model Name: {self.model_name}"
            )
            print(full_error)
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            self.error.emit(
                f"Error in translating {os.path.basename(self.input_file)}", 
                full_error
            )
    def _count_subtitles(self):
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                count = sum(1 for line in f if line.strip().isdigit())
                print(f"Successfully counted {count} subtitles")
                return count
        except Exception as e:
            print(f"Error counting subtitles: {str(e)}")
            raise
    def _setup_translator(self):
        def progress_callback(current, total):
            self.batch_progress.emit(current, total)
        return gst.GeminiSRTTranslator(
            gemini_api_key=self.api_key,
            target_language=self.target_language,
            input_file=self.input_file,
            output_file=f"{os.path.splitext(self.input_file)[0]}_translated.srt",
            batch_size=self.batch_size,
            progress_callback=progress_callback,
            model_name=self.model_name,
            description=self.description
        )
class ModelListWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key
    def run(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            models = []
            for model in genai.list_models():
                if "generateContent" in model.supported_generation_methods:
                    models.append(model.name.replace("models/", ""))
            if not models:
                self.error.emit("No available models found")
                return
            self.finished.emit(models)
        except Exception as e:
            self.error.emit(f"Error fetching models: {str(e)}")
class QuotaCheckWorker(QThread):
    finished = pyqtSignal(str, dict)
    error = pyqtSignal(str, str)
    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key
    def run(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content("Hi")
            if hasattr(model, '_last_response'):
                headers = model._last_response.headers
                quota_info = {
                    'remaining': headers.get('x-ratelimit-remaining', 'Unknown'),
                    'limit': headers.get('x-ratelimit-limit', 'Unknown'),
                    'reset': headers.get('x-ratelimit-reset', 'Unknown')
                }
            else:
                quota_info = {'status': 'API Key is valid but quota info not available'}
            self.finished.emit(self.api_key, quota_info)
        except Exception as e:
            self.error.emit(self.api_key, str(e))
class TranslatorGUI(QMainWindow):
    resized = pyqtSignal()
    def __init__(self):
        super().__init__()
        self._initialize_variables()
        self.load_settings()
        self.setAcceptDrops(True)
        self.initUI()
    def _initialize_variables(self):
        self.api_keys = []
        self.key_cycle = None
        self.active_workers = []
        self.total_subtitles = 0
        self.translated_subtitles = 0
        self.start_time = None
        self.config_file = "translator_config.json"
        self.total_files = 0
        self.completed_files = 0
        self.estimated_time = None
        self.progress_times = []
        self.request_delay = 0
        self.batch_size = 28
        self.current_file_progress = 0
        self.current_file_batches = 0
        self.total_file_batches = 0
    def initUI(self):
        main_widget = QWidget()
        main_widget.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        header_layout = QHBoxLayout()
        header_label = QLabel("Subtitle Translator")
        header_label.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(header_label, 1)
        layout.addLayout(header_layout)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(separator)
        api_group = QGroupBox("API Key Management")
        api_layout = QVBoxLayout(api_group)
        api_layout.setSpacing(10)
        api_layout.setContentsMargins(10, 15, 10, 10)
        api_keys_layout = QHBoxLayout()
        api_keys_layout.setSpacing(15)
        api_keys_layout.setContentsMargins(5, 0, 5, 0)
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(10)
        add_key_btn = QPushButton("Add Key")
        remove_key_btn = QPushButton("Remove Key")
        add_key_btn.setFixedWidth(100)
        remove_key_btn.setFixedWidth(100)
        buttons_layout.addWidget(add_key_btn)
        buttons_layout.addWidget(remove_key_btn)
        input_container = QWidget()
        input_container.setFixedWidth(int(self.width() * 0.4))
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Enter your API Key here...")
        self.api_key_input.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        input_layout.addWidget(self.api_key_input)
        self.api_key_input.setAlignment(Qt.AlignmentFlag.AlignLeft)
        api_keys_layout.addWidget(buttons_widget, 0)
        api_keys_layout.addStretch(1)
        api_keys_layout.addWidget(input_container, 0)
        api_layout.addLayout(api_keys_layout)
        self.api_keys_list = QListWidget()
        self.api_keys_list.setMaximumHeight(100)
        self.api_keys_list.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        api_layout.addWidget(self.api_keys_list)
        layout.addWidget(api_group)
        files_group = QGroupBox("File Management")
        files_layout = QVBoxLayout(files_group)
        file_btn_layout = QHBoxLayout()
        select_files_btn = QPushButton("Select Files")
        clear_files_btn = QPushButton("Clear List")
        file_btn_layout.addWidget(select_files_btn)
        file_btn_layout.addWidget(clear_files_btn)
        file_btn_layout.addStretch()
        files_layout.addLayout(file_btn_layout)
        self.files_list = QListWidget()
        self.files_list.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        files_layout.addWidget(self.files_list)
        layout.addWidget(files_group)
        settings_group = QGroupBox("Translation Settings")
        settings_layout = QVBoxLayout(settings_group)
        description_layout = QHBoxLayout()
        self.description_input = QTextEdit()
        self.description_input.setPlaceholderText("Enter additional context about the subtitle content (optional)...")
        self.description_input.setMaximumHeight(60)
        self.description_input.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        description_layout.addWidget(QLabel("Context:"))
        description_layout.addWidget(self.description_input)
        settings_layout.addLayout(description_layout)
        language_layout = QHBoxLayout()
        self.language_combo = QComboBox()
        self.language_combo.addItems([
            "Persian", "English", "Arabic", "French", "German", "Spanish", 
            "Italian", "Russian", "Chinese", "Japanese", "Korean", "Turkish"
        ])
        self.language_combo.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        language_layout.addWidget(QLabel("Target Language:"))
        language_layout.addWidget(self.language_combo)
        settings_layout.addLayout(language_layout)
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setPlaceholderText("Select a model...")
        self.model_combo.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.list_models_btn = QPushButton("List Available Models")
        self.list_models_btn.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.list_models_btn)
        settings_layout.addLayout(model_layout)
        layout.addWidget(settings_group)
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QVBoxLayout(advanced_group)
        delay_layout = QHBoxLayout()
        self.delay_input = QLineEdit()
        self.delay_input.setPlaceholderText("0")
        self.delay_input.setText("0")
        self.delay_input.setMaximumWidth(100)
        self.delay_input.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        delay_layout.addWidget(QLabel("Request Delay (seconds):"))
        delay_layout.addWidget(self.delay_input)
        delay_layout.addStretch()
        advanced_layout.addLayout(delay_layout)
        batch_layout = QHBoxLayout()
        self.batch_input = QLineEdit()
        self.batch_input.setPlaceholderText("28")
        self.batch_input.setText("28")
        self.batch_input.setMaximumWidth(100)
        self.batch_input.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        batch_layout.addWidget(QLabel("Lines per Batch:"))
        batch_layout.addWidget(self.batch_input)
        batch_layout.addStretch()
        advanced_layout.addLayout(batch_layout)
        layout.addWidget(advanced_group)
        self.translate_btn = QPushButton("Start Translation")
        self.translate_btn.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        layout.addWidget(self.translate_btn)
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        progress_layout.addWidget(self.progress_bar)
        self.time_label = QLabel("Estimated time remaining: --:--")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        progress_layout.addWidget(self.time_label)
        layout.addWidget(progress_group)
        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        log_layout.addWidget(self.log_output)
        layout.addWidget(log_group)
        select_files_btn.clicked.connect(self.select_files)
        clear_files_btn.clicked.connect(self.clear_files)
        self.translate_btn.clicked.connect(self.on_translate_click)
        self.list_models_btn.clicked.connect(self.list_available_models)
        add_key_btn.clicked.connect(self.add_api_key)
        remove_key_btn.clicked.connect(self.remove_selected_key)
        self.model_combo.currentTextChanged.connect(self.update_translate_button)
        for key in self.api_keys:
            self.api_keys_list.addItem(f"{key[:10]}...")
        if self.api_keys:
            self.key_cycle = cycle(self.api_keys)
            self.update_translate_button()
        def update_input_width():
            input_container.setFixedWidth(int(self.width() * 0.4))
        self.resized.connect(update_input_width)
        
        footer_label = QLabel("Made with ❤️ by Amo-Z")
        footer_label.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(footer_label)
    def add_api_key(self):
        key = self.api_key_input.text().strip()
        if key:
            if key not in self.api_keys:
                self.api_keys.append(key)
                self.api_keys_list.addItem(f"{key[:10]}...")
                self.api_key_input.clear()
                self.key_cycle = cycle(self.api_keys)
                self.update_translate_button()
                self.save_settings()
            else:
                QMessageBox.warning(self, "Error", "This API Key has already been added.")
    def remove_selected_key(self):
        current_item = self.api_keys_list.currentItem()
        if current_item:
            index = self.api_keys_list.row(current_item)
            if 0 <= index < len(self.api_keys):
                removed_key = self.api_keys.pop(index)
                self.api_keys_list.takeItem(index)
                self.save_settings()
                self.key_cycle = cycle(self.api_keys) if self.api_keys else None
                self.update_translate_button()
                self.log_message(f"API Key removed: {removed_key[:10]}...")
    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Subtitle Files",
            "",
            "Subtitle Files (*.srt)"
        )
        if files:
            self.add_files(files)
    def clear_files(self):
        self.files_list.clear()
        self.update_translate_button()
    def update_translate_button(self):
        can_translate = (
            self.files_list.count() > 0 and 
            len(self.api_keys) > 0 and
            bool(self.model_combo.currentText().strip())
        )
        self.translate_btn.setEnabled(can_translate)
        if can_translate:
            self.translate_btn.setToolTip("Click to start translation")
        else:
            missing = []
            if len(self.api_keys) == 0:
                missing.append("API Key")
            if self.files_list.count() == 0:
                missing.append("Files")
            if not self.model_combo.currentText().strip():
                missing.append("Model")
            self.translate_btn.setToolTip(f"Missing: {', '.join(missing)}")
    def on_translate_click(self):
        if self.translate_btn.isEnabled():
            self.start_translation()
        else:
            QMessageBox.warning(
                self,
                "Missing Requirements",
                "Please complete all required fields marked with *"
            )
    def reset_error_styles(self):
        for label in [self.api_key_label, self.files_label, self.model_label]:
            label.setStyleSheet("")
        self.api_key_input.setStyleSheet(self.normal_style)
        self.files_list.setStyleSheet(self.normal_style)
        self.model_combo.setStyleSheet(self.normal_style)
    def eventFilter(self, obj, event):
        if obj == self.translate_btn and event.type() == QEvent.Type.ToolTip:
            return False
        return super().eventFilter(obj, event)
    def update_progress(self):
        if self.total_files <= 0:
            return
            
        progress = (self.completed_files / self.total_files) * 100
        self.progress_bar.setValue(int(progress))
        
        if not self.progress_times or self.completed_files <= 0:
            return
            
        avg_time_per_file = sum(self.progress_times) / len(self.progress_times)
        remaining_files = self.total_files - self.completed_files
        estimated_seconds = avg_time_per_file * remaining_files
        estimated_time = timedelta(seconds=int(estimated_seconds))
        self.time_label.setText(f"Estimated time remaining: {str(estimated_time)}")
    def update_batch_progress(self, current_batch, total_batches):
        if total_batches == 0:
            return
            
        self.current_file_batches = current_batch
        self.total_file_batches = total_batches
        file_progress = current_batch / total_batches
        total_progress = ((self.completed_files + file_progress) / self.total_files) * 100
        self.progress_bar.setValue(int(total_progress))
        
        if current_batch > 0 and self.start_time:
            self._update_time_estimate(file_progress)
    def _update_time_estimate(self, file_progress):
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        progress_fraction = (self.completed_files + file_progress) / self.total_files
        
        if progress_fraction > 0:
            total_estimated_time = elapsed_time / progress_fraction
            remaining_time = total_estimated_time - elapsed_time
            estimated_time = timedelta(seconds=int(remaining_time))
            self.time_label.setText(f"Estimated time remaining: {str(estimated_time)}")
    def start_translation(self):
        if not self._validate_translation_params():
            return
            
        self._reset_translation_state()
        self._start_translation_workers()
    def _validate_translation_params(self):
        try:
            self.batch_size = int(self.batch_input.text() or 28)
            self.request_delay = float(self.delay_input.text() or 0)
            
            if self.batch_size < 1:
                QMessageBox.warning(self, "Error", "Batch size must be at least 1")
                return False
                
            if self.request_delay < 0:
                QMessageBox.warning(self, "Error", "Request delay must be non-negative")
                return False
                
            if not self.api_keys:
                QMessageBox.warning(self, "Error", "Please add at least one API Key.")
                return False
                
            return True
            
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid numbers for delay and batch size")
            return False
    def _reset_translation_state(self):
        self.start_time = datetime.now()
        self.total_subtitles = 0
        self.translated_subtitles = 0
        self.completed_files = 0
        self.progress_times = []
        self.total_files = self.files_list.count()
        self.progress_bar.setValue(0)
        self.time_label.setText("Estimated time remaining: calculating...")
        self.translate_btn.setEnabled(False)
    def _start_translation_workers(self):
        target_language = self.language_combo.currentText()
        selected_model = self.model_combo.currentText()
        description = self.description_input.toPlainText().strip()

        for i in range(self.files_list.count()):
            worker = TranslationWorker(
                input_file=self.files_list.item(i).text(),
                api_key=next(self.key_cycle),
                target_language=target_language,
                batch_size=self.batch_size,
                request_delay=self.request_delay,
                model_name=selected_model,
                description=description
            )
            self._setup_worker_connections(worker)
            self.active_workers.append(worker)
            worker.start()
    def _setup_worker_connections(self, worker):
        worker.finished.connect(self.on_translation_finished)
        worker.progress.connect(self.log_message)
        worker.error.connect(self.log_message)
        worker.batch_progress.connect(self.update_batch_progress)
    def on_translation_finished(self, file, subtitle_count, duration):
        filename = os.path.basename(file)
        self.translated_subtitles += subtitle_count
        self.completed_files += 1
        self.progress_times.append(duration)
        if self.total_files > 0:
            total_progress = (self.completed_files / self.total_files) * 100
            self.progress_bar.setValue(int(total_progress))
        subtitles_per_second = subtitle_count / duration if duration > 0 else 0
        log_message = (
            f"✅ Successfully translated {filename}:\n"
            f"   • Number of subtitles: {subtitle_count}\n"
            f"   • Translation time: {duration:.1f} seconds\n"
            f"   • Translation speed: {subtitles_per_second:.1f} subtitles per second"
        )
        self.log_message(log_message)
        active_count = sum(1 for worker in self.active_workers if worker.isRunning())
        if active_count == 0:
            total_duration = (datetime.now() - self.start_time).total_seconds()
            final_stats = (
                f"\n📊 Overall Statistics:\n"
                f"   • Total subtitles translated: {self.translated_subtitles}\n"
                f"   • Total time: {total_duration:.1f} seconds\n"
                f"   • Average speed: {self.translated_subtitles/total_duration:.1f} subtitles per second\n"
                f"   • Number of files translated: {self.files_list.count()}"
            )
            self.log_message(final_stats)
            self.translate_btn.setEnabled(True)
            self.time_label.setText("Translation completed!")
            self.progress_bar.setValue(100)
    def log_message(self, title, details=None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        if details:
            formatted_message = f"[{timestamp}] {title}\n{details}"
        else:
            formatted_message = f"[{timestamp}] {title}"
        print(formatted_message)
        self.log_output.append(formatted_message)
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )
    def list_available_models(self):
        if not self.api_keys:
            QMessageBox.warning(self, "Error", "Please add at least one API Key first.")
            return
        try:
            api_key = next(self.key_cycle)
            self.log_message("Fetching available models...")
            self.model_combo.clear()
            self.list_models_btn.setEnabled(False)
            self.list_models_btn.setText("Fetching...")
            worker = ModelListWorker(api_key)
            worker.finished.connect(self.update_model_list)
            worker.error.connect(self.on_model_list_error)
            worker.start()
            self.model_list_worker = worker
        except Exception as e:
            self.log_message(f"Error listing models: {str(e)}")
            self.list_models_btn.setEnabled(True)
            self.list_models_btn.setText("List Available Models")
    def update_model_list(self, models):
        try:
            self.model_combo.clear()
            self.model_combo.addItems(models)
            self.log_message("✅ Model list updated successfully")
            self.update_translate_button()
            self.list_models_btn.setEnabled(True)
            self.list_models_btn.setText("List Available Models")
        except Exception as e:
            self.log_message(f"Error updating model list: {str(e)}")
            self.list_models_btn.setEnabled(True)
            self.list_models_btn.setText("List Available Models")
    def on_model_list_error(self, error_message):
        self.log_message(error_message)
        self.list_models_btn.setEnabled(True)
        self.list_models_btn.setText("List Available Models")
    def check_quotas(self):
        if not self.api_keys:
            QMessageBox.warning(self, "Error", "No API keys available to check.")
            return
        self.log_message("Checking API quotas...")
        for api_key in self.api_keys:
            worker = QuotaCheckWorker(api_key)
            worker.finished.connect(self.update_quota_info)
            worker.error.connect(self.on_quota_check_error)
            worker.start()
            if not hasattr(self, 'quota_workers'):
                self.quota_workers = []
            self.quota_workers.append(worker)
    def update_quota_info(self, api_key, quota_info):
        for i in range(self.api_keys_list.count()):
            item = self.api_keys_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == api_key:
                display_text = [f"API Key: {api_key[:10]}..."]
                if isinstance(quota_info, dict):
                    if 'remaining' in quota_info:
                        display_text.append(f"Remaining Quota: {quota_info['remaining']}")
                    if 'limit' in quota_info:
                        display_text.append(f"Total Limit: {quota_info['limit']}")
                    if 'reset' in quota_info:
                        display_text.append(f"Reset Time: {quota_info['reset']}")
                    if 'status' in quota_info:
                        display_text.append(quota_info['status'])
                item.setText("\n".join(display_text))
                break
        self.log_message(f"✅ Quota check completed for API key: {api_key[:10]}...")
    def on_quota_check_error(self, api_key, error_message):
        for i in range(self.api_keys_list.count()):
            item = self.api_keys_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == api_key:
                item.setText(f"API Key: {api_key[:10]}...\nStatus: Error\nError: {error_message}")
                break
        self.log_message(f"❌ Error checking quota for API key {api_key[:10]}...: {error_message}")
    def update_required_stars(self):
        has_api_keys = len(self.api_keys) > 0
        self.api_key_label.setText(
            "Active API Keys" + 
            ("" if has_api_keys else " <span style='color: red'>*</span>")
        )
        has_files = self.files_list.count() > 0
        self.files_label.setText(
            "Selected Files" + 
            ("" if has_files else " <span style='color: red'>*</span>")
        )
        has_model = bool(self.model_combo.currentText().strip())
        self.model_label.setText(
            "Model" + 
            ("" if has_model else " <span style='color: red'>*</span>")
        )
        self.update_translate_button()
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            files = []
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.lower().endswith('.srt'):
                    files.append(file_path)
            self.add_files(files)
        else:
            event.ignore()
    def add_files(self, files):
        for file in files:
            if self.files_list.findItems(file, Qt.MatchFlag.MatchExactly) == []:
                self.files_list.addItem(file)
        self.update_translate_button()
    def on_model_selected(self, index):
        print(f"Model selected: {self.model_combo.currentText()}")  
        self.model_label.setText("Model")  
        self.update_translate_button()
    def save_settings(self):
        try:
            current_settings = {}
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    current_settings = json.load(f)
            current_settings.update({
                'api_keys': self.api_keys
            })
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(current_settings, f)
        except Exception as e:
            self.log_message(f"Error saving settings: {str(e)}")
    def load_settings(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    self.api_keys = settings.get('api_keys', [])
            else:
                self.save_settings()
        except Exception as e:
            self.log_message(f"Error loading settings: {str(e)}")
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    gui = TranslatorGUI()
    gui.show()
    sys.exit(app.exec())
if __name__ == '__main__':
    main()