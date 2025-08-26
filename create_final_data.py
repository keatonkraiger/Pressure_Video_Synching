import os
import json
import cv2
import numpy as np
import argparse
from pathlib import Path
import shutil


class DataProcessor:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def find_config_files(self, root_path):
        """Find all config.json files in the directory structure"""
        root = Path(root_path)
        config_files = []
        
        if (root / "config.json").exists():
            # Single folder with config
            config_files.append(root / "config.json")
        else:
            # Multiple folders, search for configs
            for item in root.iterdir():
                if item.is_dir():
                    config_path = item / "config.json"
                    if config_path.exists():
                        config_files.append(config_path)
        
        return config_files
    
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def get_video_info(self, video_path):
        """Get video information (frame count, fps, dimensions)"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        return frame_count, fps, width, height
    
    def trim_video(self, input_path, output_path, start_frame, end_frame):
        """Trim video from start_frame to end_frame"""
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Set starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_idx = start_frame
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        return frame_idx - start_frame  # Return actual frames written
    
    def trim_pressure_npy(self, input_path, output_path, start_frame, end_frame):
        """Trim pressure data from .npy file"""
        pressure_data = np.load(input_path)
        trimmed_data = pressure_data[start_frame:end_frame, ...]
        np.save(output_path, trimmed_data)
        return trimmed_data.shape[0]  # Return number of frames
    
    def create_combined_video(self, v1_path, v2_path, pressure_video_path, output_path, 
                             rgb_start, rgb_end, pressure_start, pressure_end):
        """Create combined video with V1, V2, and pressure side by side"""
        cap_v1 = cv2.VideoCapture(str(v1_path))
        cap_v2 = cv2.VideoCapture(str(v2_path))
        cap_pressure = cv2.VideoCapture(str(pressure_video_path))
        
        if not all([cap_v1.isOpened(), cap_v2.isOpened(), cap_pressure.isOpened()]):
            raise ValueError("Could not open one or more videos for combining")
        
        # Set pressure video to start frame
        cap_pressure.set(cv2.CAP_PROP_POS_FRAMES, pressure_start)
        
        # Get video properties
        fps = cap_v1.get(cv2.CAP_PROP_FPS)
        v1_w, v1_h = int(cap_v1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_v1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        v2_w, v2_h = int(cap_v2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_v2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        p_w, p_h = int(cap_pressure.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_pressure.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate output dimensions - use max height, preserve pressure aspect ratio
        output_height = max(v1_h, v2_h, p_h)
        
        # Calculate scaled dimensions preserving aspect ratios
        v1_scale = output_height / v1_h
        v2_scale = output_height / v2_h
        p_scale = output_height / p_h
        
        scaled_v1_w = int(v1_w * v1_scale)
        scaled_v2_w = int(v2_w * v2_scale)
        scaled_p_w = int(p_w * p_scale)  # Preserve pressure aspect ratio
        
        output_width = scaled_v1_w + scaled_v2_w + scaled_p_w
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
        
        frame_count = 0
        target_frames = pressure_end - pressure_start
        
        while frame_count < target_frames:
            ret_v1, frame_v1 = cap_v1.read()
            ret_v2, frame_v2 = cap_v2.read()
            ret_p, frame_p = cap_pressure.read()
            
            if not all([ret_v1, ret_v2, ret_p]):
                break
            
            # Resize frames maintaining aspect ratios
            frame_v1 = cv2.resize(frame_v1, (scaled_v1_w, output_height))
            frame_v2 = cv2.resize(frame_v2, (scaled_v2_w, output_height))
            frame_p = cv2.resize(frame_p, (scaled_p_w, output_height))  # Maintains aspect ratio
            
            # Horizontally stack the frames
            combined_frame = np.hstack([frame_v1, frame_v2, frame_p])
            out.write(combined_frame)
            frame_count += 1
        
        cap_v1.release()
        cap_v2.release()
        cap_pressure.release()
        out.release()
        
        return frame_count
    
    def find_associated_files(self, config_path):
        """Find the associated video and pressure files based on config location"""
        config_dir = config_path.parent
        base_name = config_dir.name
        
        # Look for files in the config directory
        files = {
            'rgb_v1': None,
            'rgb_v2': None,
            'pressure_npy': None,
            'pressure_video': None
        }
        
        # Common patterns for file names
        patterns = [
            (f"{base_name}_V1.mp4", 'rgb_v1'),
            (f"{base_name}_V2.mp4", 'rgb_v2'),
            (f"Original_Pressure.npy", 'pressure_npy'),
            (f"Original_Pressure.mp4", 'pressure_video'),
            (f"{base_name}.mp4", 'pressure_video')  # Alternative pressure video name
        ]
        
        for pattern, file_type in patterns:
            file_path = config_dir / pattern
            if file_path.exists():
                files[file_type] = file_path
        
        return files
    
    def process_single_config(self, config_path):
        """Process a single configuration file"""
        print(f"\nProcessing: {config_path}")
        
        # Load configuration
        config = self.load_config(config_path)
        offset = config['offset']
        
        # Find associated files in the same directory as config
        associated_files = self.find_associated_files(config_path)
        
        # Use local files instead of config paths
        rgb_video1_path = associated_files['rgb_v1']  # Primary RGB video (V1)
        rgb_video2_path = associated_files['rgb_v2']  # Secondary RGB video (V2)
        
        pressure_video_path = associated_files['pressure_video']
        
        if not rgb_video1_path or not rgb_video2_path or not pressure_video_path:
            print(f"  Error: Could not find required video files in {config_path.parent}")
            print(f"    RGB V1: {'✓' if rgb_video1_path else '✗'}")
            print(f"    RGB V2: {'✓' if rgb_video2_path else '✗'}")
            print(f"    Pressure video: {'✓' if pressure_video_path else '✗'}")
            return False
        
        # Determine output folder name
        folder_name = config_path.parent.name
        output_dir = self.save_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Output directory: {output_dir}")
        print(f"  Offset: {offset}")
        
        # Get video information
        try:
            rgb1_frame_count, rgb1_fps, _, _ = self.get_video_info(rgb_video1_path)
            rgb2_frame_count, rgb2_fps, _, _ = self.get_video_info(rgb_video2_path)
            pressure_frame_count, pressure_fps, _, _ = self.get_video_info(pressure_video_path)
        except Exception as e:
            print(f"  Error reading video info: {e}")
            return False
        
        print(f"  RGB Video: {rgb1_frame_count} frames at {rgb1_fps:.1f} fps")
        print(f"  RGB Video 2: {rgb2_frame_count} frames at {rgb2_fps:.1f} fps")
        print(f"  Pressure Video: {pressure_frame_count} frames at {pressure_fps:.1f} fps")
       
        assert rgb1_fps == pressure_fps, "RGB and Pressure videos must have different fps"
        assert rgb2_fps == pressure_fps, "RGB 2 and Pressure videos must have different fps"
        assert rgb1_fps == rgb2_fps, "RGB videos must have the same fps"

        # Calculate trimming based on offset
        if offset < 0:
            # Negative offset: trim RGB videos from the beginning
            rgb_start = abs(offset)
            pressure_start = 0
            rgb_available = rgb1_frame_count - rgb_start
            pressure_available = pressure_frame_count
        else:
            # Positive offset: trim pressure video from the beginning
            rgb_start = 0
            pressure_start = offset
            rgb_available = rgb1_frame_count
            pressure_available = pressure_frame_count - pressure_start
        
        # Determine final length (minimum of available frames)
        final_length = min(rgb_available, pressure_available)
        
        if final_length <= 0:
            print(f"  Error: No overlapping frames after applying offset {offset}")
            return False
        
        print(f"  Final length after synchronization: {final_length} frames")
        
        # Process RGB videos (V1 and V2 if available)
        rgb_end = rgb_start + final_length
        pressure_end = pressure_start + final_length
        
        try:
            # Primary RGB video (V1)
            v1_output = output_dir / "Video_V1.mp4"
            frames_written = self.trim_video(rgb_video1_path, v1_output, rgb_start, rgb_end)
            print(f"  Created Video_V1.mp4 with {frames_written} frames")
            
            # Look for V2 video (required for All.mp4)
            v2_exists = False
            if associated_files['rgb_v2']:
                v2_output = output_dir / "Video_V2.mp4"
                frames_written = self.trim_video(associated_files['rgb_v2'], v2_output, rgb_start, rgb_end)
                print(f"  Created Video_V2.mp4 with {frames_written} frames")
                v2_exists = True
            else:
                print(f"  Warning: No V2 video found - All.mp4 will not be created")
            
            # Process pressure .npy if available
            npy_exists = False
            if associated_files['pressure_npy']:
                npy_output = output_dir / "Pressure.npy"
                frames_written = self.trim_pressure_npy(associated_files['pressure_npy'], npy_output, pressure_start, pressure_end)
                print(f"  Created Pressure.npy with {frames_written} frames")
                npy_exists = True
            else:
                print(f"  Warning: No pressure .npy file found")
            
            # Create combined video only if we have V1, V2, and pressure video
            if v2_exists and pressure_video_path:
                v1_path = output_dir / "Video_V1.mp4"
                v2_path = output_dir / "Video_V2.mp4"
                
                combined_output = output_dir / "All.mp4"
                frames_written = self.create_combined_video(v1_path, v2_path, pressure_video_path, combined_output, 
                                                          rgb_start, rgb_end, pressure_start, pressure_end)
                print(f"  Created All.mp4 with {frames_written} frames")
            
            print(f"  ✓ Successfully processed {folder_name}")
            return True
            
        except Exception as e:
            print(f"  ✗ Error processing {folder_name}: {e}")
            return False
    
    def process_directory(self, input_path):
        """Process all configurations in a directory"""
        config_files = self.find_config_files(input_path)
        
        if not config_files:
            print(f"No config.json files found in {input_path}")
            return
        
        print(f"Found {len(config_files)} configuration files")
        
        successful = 0
        for config_path in config_files:
            if self.process_single_config(config_path):
                successful += 1
        
        print(f"\n✓ Successfully processed {successful}/{len(config_files)} configurations")
        print(f"Output saved to: {self.save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Process synchronized video data")
    parser.add_argument("--input_path", help="Path to directory containing config files or single config directory")
    parser.add_argument("--save_dir", help="Directory to save processed data")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without actually doing it")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        return
    
    processor = DataProcessor(args.save_dir)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be created")
        config_files = processor.find_config_files(input_path)
        print(f"Would process {len(config_files)} configuration files:")
        for config_path in config_files:
            print(f"  - {config_path}")
        return
    
    processor.process_directory(input_path)


if __name__ == "__main__":
    main()