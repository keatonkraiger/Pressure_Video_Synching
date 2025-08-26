import sys
import cv2
import json
import numpy as np
import subprocess
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSlider, QSpinBox, QGroupBox, QFormLayout, QSizePolicy,
    QMessageBox, QProgressDialog
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


class FFmpegWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, input_path, target_fps=50):
        super().__init__()
        self.input_path = input_path
        self.target_fps = target_fps
        
    def run(self):
        try:
            # Create temporary output file
            input_path = str(self.input_path)
            temp_path = input_path.replace('.mp4', '_temp_fps.mp4')
            
            self.progress.emit(f"Converting {os.path.basename(input_path)} to {self.target_fps} FPS...")
            
            # FFmpeg command to change FPS
            cmd = [
                'ffmpeg', '-i', input_path,
                '-filter:v', f'fps={self.target_fps}',
                '-y',  # Overwrite output file
                temp_path
            ]
            
            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Replace original file with converted file
                import time
                # Small delay to ensure file handles are released
                time.sleep(0.5)
                
                # On Windows, try to remove the original file first
                try:
                    if os.path.exists(input_path):
                        os.remove(input_path)
                    os.rename(temp_path, input_path)
                except OSError as e:
                    # If rename fails, try copy and delete
                    import shutil
                    shutil.copy2(temp_path, input_path)
                    os.remove(temp_path)
                
                self.finished.emit(True, f"Successfully converted to {self.target_fps} FPS")
            else:
                self.finished.emit(False, f"FFmpeg error: {result.stderr}")
                # Clean up temp file if it exists
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")


class VideoPanel:
    def __init__(self, title, parent_widget, video_path=None, is_pressure=False):
        self.title = title
        self.is_pressure = is_pressure
        self.label = QLabel(title)
        self.label.setMinimumSize(480, 360)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.play_btn = QPushButton("Play")
        self.step_btn = QPushButton("Step")
        self.restart_btn = QPushButton("Restart")

        # Display FPS and timing info
        self.fps_display = QLabel("FPS: --")
        self.time_display = QLabel("Time: --:-- / --:--")
        self.frame_idx_label = QLabel("Frame: 0")

        self.cap = None
        self.frame_idx = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.original_fps = None
        self.total_frames = 0
        self.total_duration = 0
        self.video_path = None

        self.parent = parent_widget
        self.cached_frames = []
        self.cache_limit_seconds = 15
        self._build_controls()

        if video_path:
            self.load_video(video_path)

    def _build_controls(self):
        layout = QVBoxLayout()
        layout.addWidget(self.label)

        # First row of controls
        control_layout1 = QHBoxLayout()
        control_layout1.addWidget(self.play_btn)
        control_layout1.addWidget(self.step_btn)
        control_layout1.addWidget(self.restart_btn)
        control_layout1.addStretch()
        layout.addLayout(control_layout1)

        # Second row with info
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.fps_display)
        info_layout.addWidget(self.time_display)
        info_layout.addWidget(self.frame_idx_label)
        info_layout.addStretch()
        layout.addLayout(info_layout)

        self.group_box = QGroupBox(self.title)
        self.group_box.setLayout(layout)

        self.play_btn.clicked.connect(self.toggle_play)
        self.step_btn.clicked.connect(self.step_once)
        self.restart_btn.clicked.connect(self.restart)

    def check_and_fix_fps(self):
        """Check if video is at 50 FPS and offer to fix if not"""
        if not self.video_path or not self.original_fps:
            return True
        
        target_fps = 50.0
        fps_tolerance = 1.0  # Allow small variations
        
        if abs(self.original_fps - target_fps) > fps_tolerance:
            if self.is_pressure:
                # Pressure video should always be 50 FPS
                QMessageBox.warning(
                    self.parent, 
                    "Pressure Video FPS Error", 
                    f"Pressure video is at {self.original_fps:.1f} FPS but should be 50 FPS.\n"
                    "Please fix the pressure video FPS before proceeding.\n\n"
                    "The pressure video must be exactly 50 FPS for synchronization to work correctly."
                )
                return False
            else:
                # RGB video can be converted
                msgBox = QMessageBox(self.parent)
                msgBox.setWindowTitle("RGB Video FPS Conversion")
                msgBox.setText(f"RGB video is at {self.original_fps:.1f} FPS.")
                msgBox.setInformativeText("What would you like to do?")
                
                if self.original_fps < target_fps:
                    convert_btn = msgBox.addButton("Set to 50 FPS", QMessageBox.AcceptRole)
                else:
                    convert_btn = msgBox.addButton("Set to 50 FPS", QMessageBox.AcceptRole)
                
                continue_btn = msgBox.addButton("Continue", QMessageBox.RejectRole)
                
                msgBox.exec_()
                
                if msgBox.clickedButton() == convert_btn:
                    return self.convert_fps()
                else:
                    return True  # Continue with current FPS
        
        return True  # FPS is acceptable
    
    def convert_fps(self):
        """Convert video to 50 FPS using FFmpeg"""
        try:
            # Check if ffmpeg is available
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            QMessageBox.critical(
                self.parent,
                "FFmpeg Not Found",
                "FFmpeg is not installed or not available in PATH.\n"
                "Please install FFmpeg to convert video FPS."
            )
            return False
        
        # Close the video capture to release the file handle
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Create progress dialog
        progress = QProgressDialog("Converting video FPS...", None, 0, 0, self.parent)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)  # No cancel button
        progress.show()
        QApplication.processEvents()
        
        # Create and start worker thread
        self.worker = FFmpegWorker(self.video_path, 50)
        success = [False]  # Use list to modify from inner function
        
        def on_progress(message):
            progress.setLabelText(message)
            QApplication.processEvents()
        
        def on_finished(success_flag, message):
            success[0] = success_flag
            progress.close()
            if success_flag:
                QMessageBox.information(self.parent, "Success", message)
                # Reload the video to update FPS info
                self.load_video(self.video_path)
            else:
                QMessageBox.critical(self.parent, "Conversion Failed", message)
                # Try to reload the original video even if conversion failed
                self.load_video(self.video_path)
        
        self.worker.progress.connect(on_progress)
        self.worker.finished.connect(on_finished)
        self.worker.start()
        
        # Wait for worker to finish
        self.worker.wait()
        
        return success[0]

    def load_video(self, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self.parent, f"Open {self.title}")
        if path:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                self.cap = cap
                self.video_path = path
                self.original_fps = cap.get(cv2.CAP_PROP_FPS)
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Calculate total duration
                if self.original_fps > 0:
                    self.total_duration = self.total_frames / self.original_fps
                else:
                    self.total_duration = 0
                
                # Update displays
                self.fps_display.setText(f"FPS: {self.original_fps:.1f}")
                self.update_time_display()
                
                # Check FPS and offer to fix if needed
                if not self.check_and_fix_fps():
                    # User cancelled or FPS check failed
                    if self.cap:
                        self.cap.release()
                    self.cap = None
                    self.video_path = None
                    return None
                
                self.cached_frames = []
                self._cache_initial_frames()
                self.frame_idx = 0
                return path
        return None
    
    def _cache_initial_frames(self):
        """Cache initial frames"""
        if not self.cap:
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cache_limit = int(self.original_fps * self.cache_limit_seconds) if self.original_fps > 0 else 750
        
        for _ in range(min(cache_limit, self.total_frames)):
            ret, frame = self.cap.read()
            if not ret:
                break
            self.cached_frames.append(frame)
    
    def update_time_display(self):
        """Update the time display"""
        if self.original_fps > 0:
            current_time = self.frame_idx / self.original_fps
            current_min = int(current_time // 60)
            current_sec = int(current_time % 60)
            
            total_min = int(self.total_duration // 60)
            total_sec = int(self.total_duration % 60)
            
            self.time_display.setText(f"Time: {current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}")
        else:
            self.time_display.setText("Time: --:-- / --:--")

    def get_fps_info(self):
        """Get FPS information"""
        return self.original_fps

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            # Use original FPS for playback
            if self.original_fps and self.original_fps > 0:
                interval = int(1000 / self.original_fps)
                self.timer.start(interval)
                self.play_btn.setText("Pause")

    def step_once(self):
        self.timer.stop()
        self.play_btn.setText("Play")
        self.update_frame()

    def restart(self):
        self.frame_idx = 0
        self.display_frame_by_index()
        self.update_time_display()

    def update_frame(self):
        if self.cap:
            if self.frame_idx < len(self.cached_frames):
                frame = self.cached_frames[self.frame_idx]
                self.display_frame(frame)
                self.frame_idx += 1
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
                ret, frame = self.cap.read()
                if ret:
                    self.display_frame(frame)
                    self.frame_idx += 1
                else:
                    self.timer.stop()
                    self.play_btn.setText("Play")
            
            self.frame_idx_label.setText(f"Frame: {self.frame_idx}")
            self.update_time_display()

    def display_frame_by_index(self):
        if self.frame_idx < len(self.cached_frames):
            self.display_frame(self.cached_frames[self.frame_idx])
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.label.width(), self.label.height(),
            Qt.KeepAspectRatio, Qt.FastTransformation
        )
        self.label.setPixmap(pixmap)

    def get_state(self):
        return {
            "frame_idx": self.frame_idx,
            "original_fps": self.original_fps,
            "total_frames": self.total_frames,
            "total_duration": self.total_duration
        }

    def play(self):
        if not self.timer.isActive() and self.original_fps and self.original_fps > 0:
            interval = int(1000 / self.original_fps)
            self.timer.start(interval)
            self.play_btn.setText("Pause")

    def pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")

    def set_frame_index(self, idx):
        self.frame_idx = max(0, min(idx, self.total_frames - 1))
        self.display_frame_by_index()
        self.frame_idx_label.setText(f"Frame: {self.frame_idx}")
        self.update_time_display()


class VideoSyncApp(QWidget):
    def __init__(self, rgb_video_path=None, pressure_video_path=None):
        super().__init__()
        self.setWindowTitle("Video Synchronization Tool")
        self.resize(1280, 800)

        self.rgb_video = VideoPanel("RGB Video", self, video_path=rgb_video_path)
        self.pressure_video = VideoPanel("Pressure Video", self, video_path=pressure_video_path)
        self.offset = 0

        self.offset_slider = QSlider(Qt.Horizontal)
        self.offset_slider.setRange(-1000, 1000)
        self.offset_slider.setValue(0)
        self.offset_slider.valueChanged.connect(self.set_offset)

        self.offset_box = QSpinBox()
        self.offset_box.setRange(-1000, 1000)
        self.offset_box.setValue(0)
        self.offset_box.valueChanged.connect(self.set_offset)

        self.offset_slider.valueChanged.connect(self.offset_box.setValue)
        self.offset_box.valueChanged.connect(self.offset_slider.setValue)

        self.load_rgb_btn = QPushButton("Load RGB Video")
        self.load_pressure_btn = QPushButton("Load Pressure Video")
        self.validate_fps_btn = QPushButton("Validate FPS")
        self.save_btn = QPushButton("Save Sync Config")
        self.play_both_btn = QPushButton("Play/Pause Both")
        self.test_offset_btn = QPushButton("Test Offset Playback")

        self.load_rgb_btn.clicked.connect(self.load_rgb_video)
        self.load_pressure_btn.clicked.connect(self.load_pressure_video)
        self.validate_fps_btn.clicked.connect(self.validate_fps)
        self.save_btn.clicked.connect(self.save_sync)
        self.play_both_btn.clicked.connect(self.toggle_both)
        self.test_offset_btn.clicked.connect(self.play_with_offset)

        self.rgb_video_path = rgb_video_path
        self.pressure_video_path = pressure_video_path

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        video_layout = QHBoxLayout()
        video_layout.addWidget(self.rgb_video.group_box)
        video_layout.addWidget(self.pressure_video.group_box)

        sync_layout = QFormLayout()
        sync_layout.addRow("Offset (frames):", self.offset_slider)
        sync_layout.addRow("", self.offset_box)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.load_rgb_btn)
        control_layout.addWidget(self.load_pressure_btn)
        control_layout.addWidget(self.validate_fps_btn)
        control_layout.addStretch()
        control_layout.addWidget(self.play_both_btn)
        control_layout.addWidget(self.test_offset_btn)
        control_layout.addWidget(self.save_btn)

        layout.addLayout(video_layout)
        layout.addLayout(sync_layout)
        layout.addLayout(control_layout)
        self.setLayout(layout)

    def set_offset(self, val):
        self.offset = val

    def load_rgb_video(self):
        self.rgb_video_path = self.rgb_video.load_video()

    def load_pressure_video(self):
        self.pressure_video_path = self.pressure_video.load_video()

    def validate_fps(self):
        """Manually validate FPS of both videos"""
        rgb_valid = True
        pressure_valid = True
        
        if self.rgb_video.cap:
            rgb_valid = self.rgb_video.check_and_fix_fps()
        
        if self.pressure_video.cap:
            pressure_valid = self.pressure_video.check_and_fix_fps()
        
        if rgb_valid and pressure_valid:
            QMessageBox.information(self, "FPS Validation", "Both videos have valid FPS (50 FPS).")
        elif not rgb_valid:
            QMessageBox.warning(self, "FPS Validation", "RGB video FPS validation failed.")
        elif not pressure_valid:
            QMessageBox.warning(self, "FPS Validation", "Pressure video FPS validation failed.")

    def toggle_both(self):
        if self.rgb_video.timer.isActive() or self.pressure_video.timer.isActive():
            self.rgb_video.pause()
            self.pressure_video.pause()
        else:
            self.rgb_video.play()
            self.pressure_video.play()

    def play_with_offset(self):
        self.rgb_video.pause()
        self.pressure_video.pause()
        
        rgb_start = max(0, -self.offset)
        pressure_start = max(0, self.offset)

        self.rgb_video.set_frame_index(rgb_start)
        self.pressure_video.set_frame_index(pressure_start)

        self.rgb_video.play()
        self.pressure_video.play()

    def save_sync(self):
        if not self.rgb_video_path or not self.pressure_video_path:
            QMessageBox.warning(self, "Error", "Load both videos before saving sync.")
            return

        config = {
            "rgb_video_path": self.rgb_video_path,
            "pressure_video_path": self.pressure_video_path,
            "offset": self.offset,
            "rgb_video_fps": self.rgb_video.get_fps_info(),
            "pressure_fps": self.pressure_video.get_fps_info()
        }

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Config", filter="JSON Files (*.json)")
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(config, f, indent=2)
            QMessageBox.information(self, "Success", f"Saved sync config to {save_path}")

    def load_sync_config(self):
        """Load a previously saved sync configuration"""
        config_path, _ = QFileDialog.getOpenFileName(self, "Load Sync Config", filter="JSON Files (*.json)")
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Load videos if paths exist
                if config.get("rgb_video_path") and os.path.exists(config["rgb_video_path"]):
                    self.rgb_video_path = self.rgb_video.load_video(config["rgb_video_path"])
                
                if config.get("pressure_video_path") and os.path.exists(config["pressure_video_path"]):
                    self.pressure_video_path = self.pressure_video.load_video(config["pressure_video_path"])
                
                # Set offset
                if "offset" in config:
                    self.offset = config["offset"]
                    self.offset_slider.setValue(self.offset)
                    self.offset_box.setValue(self.offset)
                
                QMessageBox.information(self, "Success", f"Loaded sync config from {config_path}")
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load config: {str(e)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, help="Path to RGB video")
    parser.add_argument("--pressure", type=str, help="Path to pressure video")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = VideoSyncApp(rgb_video_path=args.rgb, pressure_video_path=args.pressure)
    win.show()
    sys.exit(app.exec_())