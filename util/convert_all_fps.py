# import numpy as np
# import os
# import re
# import subprocess
# import tempfile
# import shutil
# import cv2
# root_dir = r"Data/Spencer"

# pattern = re.compile(r"^Original_Pressure.mp4$", re.IGNORECASE)

# def get_video_info( video_path):
#     """Get video information (frame count, fps, dimensions)"""
#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         raise ValueError(f"Could not open video: {video_path}")
    
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     cap.release()
#     return frame_count, fps

# def get_fps(video_path):
#     """Return the fps of a video using ffprobe."""
#     try:
#         result = subprocess.run(
#             [
#                 "ffprobe", "-v", "error",
#                 "-select_streams", "v:0",
#                 "-show_entries", "stream=r_frame_rate",
#                 "-of", "default=noprint_wrappers=1:nokey=1",
#                 video_path
#             ],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True,
#             check=True
#         )
#         num, den = result.stdout.strip().split('/')
#         return float(num) / float(den)
#     except Exception as e:
#         print(f"Error reading fps for {video_path}: {e}")
#         return None

# def process_video(video_path, out_path):
#     """Convert video to 50 fps if needed."""
#     fps = get_fps(video_path)
#     if fps is None:
#         return
    
#     if abs(fps - 50) < 0.01:  # already 50 fps
#         print(f"Skipping {video_path} (already 50 fps)")
#         return

#     # create a temporary output file in the same directory
    
#     tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=os.path.dirname(out_path))
#     os.close(tmp_fd)  # we only need the path, not the open handle
    
#     try:
#         print(f"Converting {video_path} ({fps:.2f} fps → 50 fps)")
#         om_id = video_path.split('/')[-2]
#         rgb_video_path = video_path.replace("Original_Pressure", f"{om_id}_V1")
#         frame_count, rgb_fps = get_video_info(rgb_video_path)
#         # Load the pressure
#         pressure_path = video_path.replace("mp4", 'npy')
#         pressure = np.load(pressure_path)
#         pressure_frames = pressure.shape[0] 
        
#         if pressure_frames > (frame_count * 1.5):
#             print("The pressure seems to be at 100 fps, not 50")
#             # Get user input if they want to downsample and save the pressure at 50 fps
#             user_input = input("Do you want to downsample the pressure to 50 fps? (y/n): ")
#             if user_input.lower() == 'y':
#                 pressure = pressure.reshape((pressure_frames, -1))
#                 pressure = pressure[::2,...]  # Downsample by taking every second frame
#                 np.save(pressure_path, pressure)
#                 print(f"Downsampled pressure saved to {pressure_path}")
                
#         subprocess.run(
#             [
#                 "ffmpeg", "-y", "-i", video_path,
#                 "-filter:v", "fps=50",
#                 "-c:a", "copy",  # keep audio unchanged
#                 # Use more CPU
#                 "-preset", "veryfast",
#                 tmp_path
#             ],
#             check=True
#         )
#         # replace target with the temp file
#         shutil.move(tmp_path, out_path)

#     except subprocess.CalledProcessError as e:
#         print(f"FFmpeg failed for {video_path}: {e}")
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)

# for subdir, _, files in os.walk(root_dir):
#     for file in files:
#         if pattern.match(file):
#             in_path = os.path.join(subdir, file)
#             out_name = file
#             out_path = os.path.join(subdir, out_name)
#             process_video(in_path, out_path)
import os
import re
import subprocess

root_dir = r"Data/Spencer"

# regex for OM<num>_V1.mp4 or OM<num>_V2.mp4
pattern = re.compile(r"^OM\d+_V[12]\.mp4$", re.IGNORECASE)

def get_fps(video_path):
    """Return the fps of a video using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        num, den = result.stdout.strip().split('/')
        return float(num) / float(den)
    except Exception as e:
        print(f"Error reading fps for {video_path}: {e}")
        return None

def process_video(video_path, out_path):
    """Convert video to 50 fps if needed."""
    fps = get_fps(video_path)
    if fps is None:
        return
    
    if abs(fps - 50) < 0.01:  # already 50 fps
        print(f"Skipping {video_path} (already 50 fps)")
        return
    
    print(f"Converting {video_path} ({fps:.2f} fps → 50 fps)")
    subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-filter:v", "fps=50",
            "-c:a", "copy",  # keep audio unchanged
            out_path
        ],
        check=True
    )

for subdir, _, files in os.walk(root_dir):
    for file in files:
        if pattern.match(file):
            in_path = os.path.join(subdir, file)
            out_name = file.split('_')[1]  # e.g. V1.mp4
            out_path = os.path.join(subdir, out_name)
            process_video(in_path, out_path)
