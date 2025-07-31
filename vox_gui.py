#!/usr/bin/env python3

# Vox: Voice Transcription Tool using Deepgram API
# Usage: ./vox_gui.py
# Dependencies: pip install sounddevice numpy requests pyperclip scipy pyqt5
# Setup: Run 'chmod +x vox_gui.py' and enter Deepgram API key in the GUI

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyperclip
import requests
import scipy.io.wavfile as wavfile
import sounddevice as sd
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QLabel, QLineEdit, QMainWindow,
                             QMenu, QMessageBox, QPushButton, QSystemTrayIcon,
                             QVBoxLayout, QWidget)

# Configuration defaults
SAMPLE_RATE = 16000  # Hz
CHANNELS = 1  # Mono
AUDIO_FORMAT = "wav"
PID_FILE = Path.home() / ".recordpid"
AUDIO_FILE_TRACKER = Path.home() / ".recordaudio"
BASE_FILE = Path.home() / ".vox" / "recording"

# Global state
recording_process = None
audio_data = []
stop_event = threading.Event()
debug_mode = False


class VoxWindow(QMainWindow):
    """Main window for Vox GUI."""

    def __init__(self, app):
        """Initialize the window."""
        super().__init__()
        self.app = app  # Store QApplication instance for shutdown
        self.setWindowTitle("Vox")
        self.setFixedSize(450, 375)
        # Initialize configuration variables
        self.deepgram_api_key = ""
        self.max_duration = 120
        self.deepgram_model = "nova-3"
        self.transcript = ""
        self.setup_ui()
        self.setup_tray()
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def setup_ui(self):
        """Set up the user interface."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        font = QFont()
        font.setPointSize(12)

        self.api_key_label = QLabel("Deepgram API Key:")
        self.api_key_label.setFont(font)
        self.layout.addWidget(self.api_key_label)
        self.api_key_input = QLineEdit()
        self.api_key_input.setFont(font)
        self.api_key_input.setPlaceholderText("Enter your Deepgram API key")
        self.api_key_input.textChanged.connect(self.update_api_key)
        self.layout.addWidget(self.api_key_input)

        self.duration_label = QLabel("Max Duration (seconds):")
        self.duration_label.setFont(font)
        self.layout.addWidget(self.duration_label)
        self.duration_input = QLineEdit(str(self.max_duration))
        self.duration_input.setFont(font)
        self.duration_input.setPlaceholderText("Enter max recording duration")
        self.duration_input.textChanged.connect(self.update_duration)
        self.layout.addWidget(self.duration_input)

        self.model_label = QLabel("Deepgram Model:")
        self.model_label.setFont(font)
        self.layout.addWidget(self.model_label)
        self.model_input = QLineEdit(self.deepgram_model)
        self.model_input.setFont(font)
        self.model_input.setPlaceholderText("Enter Deepgram model (e.g., nova-3)")
        self.model_input.textChanged.connect(self.update_model)
        self.layout.addWidget(self.model_input)

        self.status_label = QLabel("Ready to record")
        self.status_label.setFont(font)
        self.layout.addWidget(self.status_label)

        self.time_label = QLabel("Recording time: 00:00")
        self.time_label.setFont(font)
        self.layout.addWidget(self.time_label)

        self.start_button = QPushButton("Start Recording")
        self.start_button.setFont(font)
        self.start_button.clicked.connect(self.start_recording)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.setFont(font)
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        self.layout.addWidget(self.stop_button)

        self.copy_button = QPushButton("Copy Transcript")
        self.copy_button.setFont(font)
        self.copy_button.clicked.connect(self.copy_transcript)
        self.copy_button.setEnabled(False)
        self.layout.addWidget(self.copy_button)

        self.debug_button = QPushButton("Enable Debug Mode")
        self.debug_button.setFont(font)
        self.debug_button.clicked.connect(self.toggle_debug)
        self.layout.addWidget(self.debug_button)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_counter)

    def setup_tray(self):
        """Set up the system tray icon."""
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setToolTip("Vox Voice Transcription")
        self.tray_icon.setIcon(self.style().standardIcon(self.style().SP_MediaVolume))
        self.tray_icon.activated.connect(self.tray_activated)

        tray_menu = QMenu()
        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(self.quit_application)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def closeEvent(self, event):
        """Handle close event."""
        event.ignore()
        self.hide()
        self.tray_icon.showMessage(
            "Vox",
            "Minimized to system tray. Click to restore, right-click to quit.",
            QSystemTrayIcon.Information,
            2000,
        )

    def tray_activated(self, reason):
        """Handle system tray icon activation."""
        if reason == QSystemTrayIcon.Trigger:
            if self.isVisible():
                self.hide()
                self.tray_icon.showMessage(
                    "Vox",
                    "Minimized to system tray. Click to restore.",
                    QSystemTrayIcon.Information,
                    2000,
                )
            else:
                self.showNormal()

    def signal_handler(self, signum, frame):
        """Handle signals."""
        self.status_label.setText("Shutting down...")
        self.cleanup()
        self.quit_application()

    def quit_application(self):
        """Quit the application."""
        self.cleanup()
        self.tray_icon.hide()
        self.app.quit()

    def update_api_key(self, text):
        """Update the Deepgram API key."""
        self.deepgram_api_key = text

    def update_duration(self, text):
        """Update the maximum recording duration."""
        try:
            duration = int(text)
            if duration <= 0:
                raise ValueError
            self.max_duration = duration
        except ValueError:
            self.status_label.setText("Error: Max duration must be a positive integer")
            self.max_duration = 120
            self.duration_input.setText(str(self.max_duration))

    def update_model(self, text):
        """Update the Deepgram model."""
        self.deepgram_model = text if text else "nova-3"

    def update_counter(self):
        """Update the recording counter."""
        elapsed = int(time.time() - self.start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        self.time_label.setText(f"Recording time: {minutes:02d}:{seconds:02d}")

    def copy_transcript(self):
        """Copy the transcript to the clipboard."""
        if self.transcript:
            pyperclip.copy(self.transcript)
            self.status_label.setText("Transcript copied to clipboard")
        else:
            self.status_label.setText("No transcript available to copy")

    def callback(self, indata, frames, time, status):
        """Audio stream callback."""
        if status:
            self.status_label.setText(f"Error: {status}")
        if not stop_event.is_set():
            audio_data.append(indata.copy())

    def cleanup(self):
        """Clean up resources."""
        global recording_process
        if recording_process:
            recording_process.stop()
            recording_process.close()
            recording_process = None
        stop_event.set()
        PID_FILE.unlink(missing_ok=True)
        AUDIO_FILE_TRACKER.unlink(missing_ok=True)

    def start_recording(self):
        """Start audio recording."""
        if not self.deepgram_api_key:
            QMessageBox.critical(
                self,
                "Error",
                "Deepgram API key is required.\n"
                "Enter it in the input field or get one from https://console.deepgram.com",
            )
            return
        global recording_process
        self.cleanup()
        BASE_FILE.parent.mkdir(parents=True, exist_ok=True)
        audio_file = f"{BASE_FILE}_{time.strftime('%Y%m%d_%H%M%S')}.{AUDIO_FORMAT}"
        with open(AUDIO_FILE_TRACKER, "w") as f:
            f.write(audio_file)
        self.status_label.setText(f"Recording to {audio_file}...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.copy_button.setEnabled(False)
        self.debug_button.setEnabled(False)
        self.api_key_input.setEnabled(False)
        self.duration_input.setEnabled(False)
        self.model_input.setEnabled(False)
        audio_data.clear()
        stop_event.clear()
        recording_process = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, callback=self.callback
        )
        recording_process.start()
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))
        self.start_time = time.time()
        self.timer.start(100)

        def recording_thread():
            stop_event.wait(timeout=self.max_duration)
            if not stop_event.is_set():
                stop_event.set()
                self.stop_recording()

        threading.Thread(target=recording_thread, daemon=True).start()

    def stop_recording(self):
        """Stop audio recording."""
        global recording_process
        self.status_label.setText("Stopping recording...")
        self.timer.stop()
        if PID_FILE.exists() and recording_process:
            stop_event.set()
            audio_file = AUDIO_FILE_TRACKER.read_text().strip()
            if audio_data:
                wavfile.write(audio_file, SAMPLE_RATE, np.concatenate(audio_data))
                self.cleanup()
                self.status_label.setText("Transcribing...")
                self.central_widget.setEnabled(False)
                self.transcribe(audio_file)
                self.central_widget.setEnabled(True)
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.copy_button.setEnabled(bool(self.transcript))
                self.debug_button.setEnabled(True)
                self.api_key_input.setEnabled(True)
                self.duration_input.setEnabled(True)
                self.model_input.setEnabled(True)
            else:
                self.status_label.setText("No audio data recorded.")
                self.cleanup()
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.copy_button.setEnabled(False)
                self.debug_button.setEnabled(True)
                self.api_key_input.setEnabled(True)
                self.duration_input.setEnabled(True)
                self.model_input.setEnabled(True)
        else:
            self.status_label.setText("No recording to stop.")
            self.cleanup()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.copy_button.setEnabled(False)
            self.debug_button.setEnabled(True)
            self.api_key_input.setEnabled(True)
            self.duration_input.setEnabled(True)
            self.model_input.setEnabled(True)

    def transcribe(self, audio_file):
        """Transcribe audio file using Deepgram API."""
        if not audio_file or not Path(audio_file).exists():
            self.status_label.setText("Error: Audio file missing or not set")
            self.transcript = ""
            return
        error_file = f"{audio_file}.error"
        json_file = f"{audio_file}.json"
        txt_file = f"{audio_file}.txt"
        try:
            with open(audio_file, "rb") as f:
                response = requests.post(
                    f"https://api.deepgram.com/v1/listen?model={self.deepgram_model}&smart_format=true",
                    headers={
                        "Authorization": f"Token {self.deepgram_api_key}",
                        "Content-Type": f"audio/{AUDIO_FORMAT}",
                    },
                    data=f,
                )
            response.raise_for_status()
            with open(json_file, "w") as f:
                json.dump(response.json(), f)
            if debug_mode:
                self.status_label.setText(f"DEBUG: Response saved in {json_file}")
                print(
                    f"DEBUG: Full Deepgram API response saved in {json_file}:",
                    file=sys.stderr,
                )
                print(json.dumps(response.json(), indent=2), file=sys.stderr)
            self.transcript = response.json()["results"]["channels"][0]["alternatives"][
                0
            ]["transcript"]
            with open(txt_file, "w") as f:
                f.write(self.transcript.rstrip())
            self.status_label.setText(
                f"Transcript saved to {txt_file}. Click Copy to copy."
            )
            Path(error_file).unlink(missing_ok=True)
        except requests.RequestException as e:
            with open(error_file, "w") as f:
                f.write(str(e))
            self.status_label.setText(f"Error: Transcription failed. See {error_file}")
            self.transcript = ""
        except (KeyError, IndexError):
            self.status_label.setText(
                f"Error: Failed to extract transcript. Check {json_file}"
            )
            self.transcript = ""

    def toggle_debug(self):
        """Toggle debug mode."""
        global debug_mode
        debug_mode = not debug_mode
        self.debug_button.setText(
            "Disable Debug Mode" if debug_mode else "Enable Debug Mode"
        )
        self.status_label.setText(
            f"Debug mode {'enabled' if debug_mode else 'disabled'}"
        )


def main():
    """Main function."""
    app = QApplication(sys.argv)
    window = VoxWindow(app)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
