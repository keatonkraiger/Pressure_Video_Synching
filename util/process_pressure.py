import subprocess
from glob import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from pathlib import Path
from scipy.ndimage import gaussian_filter, convolve
import cv2
import shutil


def _get_video_info(video_path: Path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate,nb_frames',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    output = subprocess.run(cmd, capture_output=True, text=True)
    if output.returncode != 0:
        print(f"Error getting video info for {video_path}: {output.stderr}")
        return None, 0
    fps, num_frames = output.stdout.splitlines()
    if '/' in fps:
        fps = eval(fps)
    return int(fps), int(num_frames)


def apply_essential_preprocessing(data, clip_sensor_max=True, remove_temporal_outliers=True, 
                                remove_isolated=True, gaussian_smooth=True):
    processed = np.copy(data)
    stats = {'sensor_max_exceeded': 0, 'temporal_outliers_removed': 0, 'isolated_pixels_removed': 0}
    
    # Keep track of NaN locations (sensors that don't exist)
    nan_mask = np.isnan(data)
    
    if clip_sensor_max:
        sensor_max_mask = processed > 862
        stats['sensor_max_exceeded'] = np.sum(sensor_max_mask)
        processed[sensor_max_mask] = 0
    
    if gaussian_smooth:
        for t in range(processed.shape[0]):
            frame = processed[t]
            valid_mask = ~np.isnan(frame) & (frame > 0)
            if np.any(valid_mask):
                # Only smooth valid (non-NaN, non-zero) regions
                smoothed = gaussian_filter(np.nan_to_num(frame), sigma=0.5)
                processed[t] = np.where(valid_mask, smoothed, frame)
    
    if remove_temporal_outliers:
        window_size = 5
        threshold_std = 3.0
        
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                # Skip if this location is always NaN (sensor doesn't exist)
                if np.all(nan_mask[:, i, j]):
                    continue
                    
                pixel_series = processed[:, i, j]
                valid_indices = ~np.isnan(pixel_series) & (pixel_series > 0)
                
                if np.sum(valid_indices) < 3:
                    continue
                    
                for t in range(len(pixel_series)):
                    if np.isnan(pixel_series[t]) or pixel_series[t] == 0:
                        continue
                        
                    start_idx = max(0, t - window_size // 2)
                    end_idx = min(len(pixel_series), t + window_size // 2 + 1)
                    window = pixel_series[start_idx:end_idx]
                    
                    valid_window = window[~np.isnan(window) & (window > 0)]
                    if len(valid_window) < 3:
                        continue
                        
                    window_mean = np.mean(valid_window)
                    window_std = np.std(valid_window)
                    
                    if window_std > 0 and abs(pixel_series[t] - window_mean) > threshold_std * window_std:
                        processed[t, i, j] = 0
                        stats['temporal_outliers_removed'] += 1
    
    if remove_isolated:
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0
        
        for t in range(processed.shape[0]):
            frame = processed[t]
            valid_mask = ~np.isnan(frame)
            active_mask = valid_mask & (frame > 0)
            
            if np.any(active_mask):
                neighbor_count = convolve(active_mask.astype(int), kernel, mode='constant')
                isolated_mask = (neighbor_count < 1) & active_mask
                stats['isolated_pixels_removed'] += np.sum(isolated_mask)
                processed[t][isolated_mask] = 0
    
    # Restore NaNs where sensors don't exist
    processed[nan_mask] = np.nan
    
    return processed, stats


def preprocess_pressure_block(pressure_block, **kwargs):
    if pressure_block.ndim == 4 and pressure_block.shape[-1] == 2:
        left = pressure_block[..., 0]
        right = pressure_block[..., 1]
        original_raw = np.concatenate([left, right], axis=2)
    elif pressure_block.ndim == 3:
        original_raw = pressure_block
    else:
        raise ValueError(f"Unexpected pressure data shape: {pressure_block.shape}")

    # For original display: clip to [0, 862], preserve NaNs
    original_display = np.where(np.isnan(original_raw), np.nan, np.clip(original_raw, 0, 862))
    
    T, H, W2 = original_raw.shape
    if pressure_block.ndim == 4:
        W = W2 // 2
        left_orig = original_raw[..., :W]
        right_orig = original_raw[..., W:]
        
        left_processed, left_stats = apply_essential_preprocessing(left_orig, **kwargs)
        right_processed, right_stats = apply_essential_preprocessing(right_orig, **kwargs)
        
        processed = np.concatenate([left_processed, right_processed], axis=2)
    else:
        processed, _ = apply_essential_preprocessing(original_raw, **kwargs)
    
    # Calculate difference only where both original and processed are valid (not NaN)
    # Set difference to 0 where data is NaN (so NaN regions don't show as differences)
    valid_mask = ~np.isnan(original_display) & ~np.isnan(processed)
    difference = np.zeros_like(original_display)
    difference[valid_mask] = np.abs(original_display[valid_mask] - processed[valid_mask])
    # Set NaN regions in difference to NaN so they're transparent
    difference[np.isnan(original_display) | np.isnan(processed)] = np.nan
    
    # Determine value ranges based on actual data
    # Original: use 95th percentile of valid data for better visibility
    valid_orig = original_display[~np.isnan(original_display) & (original_display > 0)]
    if len(valid_orig) > 0:
        orig_max = max(100, np.percentile(valid_orig, 95))
    else:
        orig_max = 862
        
    # Processed: use 95th percentile of PROCESSED data, not original scale
    valid_proc = processed[~np.isnan(processed) & (processed > 0)]
    if len(valid_proc) > 0:
        proc_max = max(100, np.percentile(valid_proc, 95))
        print(f"    • Processed data max: {np.max(valid_proc):.1f} kPa, using 95th percentile: {proc_max:.1f} kPa")
    else:
        proc_max = 100
    
    # Difference: use 95th percentile of actual differences
    valid_diff = difference[~np.isnan(difference) & (difference > 0)]
    if len(valid_diff) > 0:
        diff_max = max(10, np.percentile(valid_diff, 95))
    else:
        diff_max = 50
    
    value_ranges = {
        'original': (0, orig_max),
        'processed': (0, proc_max), 
        'difference': (0, proc_max)  # Use processed max for difference scale
    }
    
    return original_display, processed, difference, value_ranges


def save_pressure_video(original, processed, difference, value_ranges, output_path, fps=50):
    N = original.shape[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # Remove the title completely
    
    panels = [
        (original, "Original\nPressure", 'jet', value_ranges['original']),
        (processed, "Processed\nPressure", 'jet', value_ranges['processed']),
        (difference, "Difference\n(Removed)", 'Reds', value_ranges['difference'])
    ]
    
    ims = []
    
    for i, (data, panel_title, cmap, (vmin, vmax)) in enumerate(panels):
        ax = axes[i]
        ax.set_title(panel_title, fontsize=10, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Create masked array to handle NaNs properly (NaNs will be transparent)
        masked_data = np.ma.masked_invalid(data[0])
        
        # For difference panel, also mask zeros to avoid confusion with NaNs
        if i == 2:  # difference panel
            masked_data = np.ma.masked_where((data[0] == 0) | np.isnan(data[0]), data[0])
            
        im = ax.imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        ims.append(im)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('kPa' if i < 2 else 'Δ kPa', rotation=270, labelpad=15, fontsize=8)
        cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    
    def update_pressure(frame_idx):
        for i, (data, _, _, _) in enumerate(panels):
            if i == 2:  # difference panel - mask both zeros and NaNs
                masked_data = np.ma.masked_where((data[frame_idx] == 0) | np.isnan(data[frame_idx]), data[frame_idx])
            else:
                masked_data = np.ma.masked_invalid(data[frame_idx])
            ims[i].set_data(masked_data)
        return ims
    
    ani = animation.FuncAnimation(fig, update_pressure, frames=N, blit=True, interval=1000/fps)
    writer = animation.FFMpegWriter(fps=fps, bitrate=8000, extra_args=['-vcodec', 'libx264'])
    ani.save(output_path, writer=writer, dpi=100)
    plt.close()


def create_combined_video(v1_path, v2_path, pressure_video_path, output_path, fps=50, debug_frames=None):
    cap_v1 = cv2.VideoCapture(str(v1_path))
    cap_v2 = cv2.VideoCapture(str(v2_path))
    cap_pressure = cv2.VideoCapture(str(pressure_video_path))
    
    if not all([cap_v1.isOpened(), cap_v2.isOpened(), cap_pressure.isOpened()]):
        raise ValueError("Could not open one or more videos for combining")
    
    v1_w, v1_h = int(cap_v1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_v1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    v2_w, v2_h = int(cap_v2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_v2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    p_w, p_h = int(cap_pressure.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_pressure.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_height = max(v1_h, v2_h, p_h)
    
    v1_scale = output_height / v1_h if v1_h > 0 else 1
    v2_scale = output_height / v2_h if v2_h > 0 else 1
    p_scale = output_height / p_h if p_h > 0 else 1
    
    scaled_v1_w = int(v1_w * v1_scale)
    scaled_v2_w = int(v2_w * v2_scale)
    scaled_p_w = int(p_w * p_scale)
    
    output_width = scaled_v1_w + scaled_v2_w + scaled_p_w
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
    
    frame_count = 0
    max_frames = min(
        int(cap_v1.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap_v2.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap_pressure.get(cv2.CAP_PROP_FRAME_COUNT))
    )
    
    if debug_frames:
        max_frames = min(max_frames, debug_frames)
    
    while frame_count < max_frames:
        ret_v1, frame_v1 = cap_v1.read()
        ret_v2, frame_v2 = cap_v2.read()
        ret_p, frame_p = cap_pressure.read()
        
        if not all([ret_v1, ret_v2, ret_p]):
            break
        
        frame_v1 = cv2.resize(frame_v1, (scaled_v1_w, output_height))
        frame_v2 = cv2.resize(frame_v2, (scaled_v2_w, output_height))
        frame_p = cv2.resize(frame_p, (scaled_p_w, output_height))
        
        combined_frame = np.hstack([frame_v1, frame_v2, frame_p])
        out.write(combined_frame)
        frame_count += 1
    
    cap_v1.release()
    cap_v2.release()
    cap_pressure.release()
    out.release()
    
    return frame_count


class DatasetVisualizer:
    def __init__(self, root_dir, person_name):
        self.root_dir = Path(root_dir)
        self.person_name = person_name
        
        self.om_titles = {
            'OM1': 'Circular Walking', 'OM2': 'Straight Walking', 'OM3': 'Lateral Walking',
            'OM4': 'Single Leg Stance', 'OM5': 'Calf Raise', 'OM6': 'Squat',
            'OM7': 'Lunges', 'OM8': 'Leg Kick', 'OM9': 'Push and Pull',
            'OM10': 'Throwing', 'OM11': 'Warm Up'
        }
    
    def load_pressure_data(self, om_folder):
        npy_files = list(om_folder.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy file found in {om_folder}")
        return np.load(npy_files[0])

    def create_visualization(self, om_folder, output_dir=None, fps=50, debug_frames=None, **processing_kwargs):
        print(f"Processing {om_folder.name}...")
        
        processing_flags = []
        if processing_kwargs.get('clip_sensor_max', True):
            processing_flags.append("clip")
        if processing_kwargs.get('remove_temporal_outliers', True):
            processing_flags.append("temporal")
        if processing_kwargs.get('remove_isolated', True):
            processing_flags.append("isolated")
        if processing_kwargs.get('gaussian_smooth', True):
            processing_flags.append("smooth")
        
        processing_suffix = "_" + "-".join(processing_flags) if processing_flags else "_raw"
        
        if output_dir is None:
            output_dir = self.root_dir / "visualizations" / self.person_name
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        temp_dir = output_dir / "_temp" / om_folder.name
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        v1_path = om_folder / "Video_V1.mp4"
        v2_path = om_folder / "Video_V2.mp4"
        if not v1_path.exists() or not v2_path.exists():
            print(f"Video files not found for {om_folder.name}")
            return

        pressure_data = self.load_pressure_data(om_folder)
        
        v1_fps, v1_frame_count = _get_video_info(v1_path)
        v2_fps, v2_frame_count = _get_video_info(v2_path)
        pressure_frame_count = pressure_data.shape[0]

        assert v1_frame_count == v2_frame_count == pressure_frame_count
        assert v1_fps == v2_fps == fps

        original, processed, difference, value_ranges = preprocess_pressure_block(pressure_data, **processing_kwargs)
        
        om_title = self.om_titles.get(om_folder.name, om_folder.name)
        
        pressure_video_path = temp_dir / f"pressure{processing_suffix}.mp4"
        save_pressure_video(original, processed, difference, value_ranges, 
                          pressure_video_path, fps=fps)
        
        final_output_path = output_dir / f"{om_folder.name}{processing_suffix}.mp4"
        frame_count = create_combined_video(v1_path, v2_path, pressure_video_path, 
                                         final_output_path, fps=fps, debug_frames=debug_frames)
        
        shutil.rmtree(temp_dir)
        print(f"[OK] {final_output_path} ({frame_count} frames)")

    def process_all_om_folders(self, output_dir=None, fps=30, debug_frames=None, **processing_kwargs):
        person_root = self.root_dir / self.person_name
        om_folders = [f for f in person_root.iterdir() if f.is_dir() and f.name.startswith("OM")]
        
        if not om_folders:
            print(f"[WARN] No OM folders found in {person_root}")
            return
        
        print(f"[INFO] Found {len(om_folders)} OM folders")
        for om_folder in sorted(om_folders):
            try:
                self.create_visualization(om_folder, output_dir, fps, debug_frames, **processing_kwargs)
            except Exception as e:
                print(f"[ERROR] Failed on {om_folder.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Create streamlined pressure visualizations')
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--person_name', required=True)
    parser.add_argument('--output_dir')
    parser.add_argument('--om_folder')
    parser.add_argument('--fps', type=int, default=50)
    parser.add_argument('--debug_frames', type=int)
    
    parser.add_argument('--no-clip-sensor-max', action='store_false', dest='clip_sensor_max')
    parser.add_argument('--no-temporal-outliers', action='store_false', dest='remove_temporal_outliers')
    parser.add_argument('--no-isolated-pixels', action='store_false', dest='remove_isolated')
    parser.add_argument('--no-gaussian-smooth', action='store_false', dest='gaussian_smooth')
    
    parser.set_defaults(clip_sensor_max=True, remove_temporal_outliers=True, remove_isolated=True, gaussian_smooth=True)
    
    args = parser.parse_args()
    
    processing_kwargs = {
        'clip_sensor_max': args.clip_sensor_max,
        'remove_temporal_outliers': args.remove_temporal_outliers,
        'remove_isolated': args.remove_isolated,
        'gaussian_smooth': args.gaussian_smooth
    }
    
    visualizer = DatasetVisualizer(args.root_dir, args.person_name)
    
    if args.om_folder:
        om_path = Path(args.root_dir) / args.person_name / args.om_folder
        visualizer.create_visualization(om_path, args.output_dir, args.fps, args.debug_frames, **processing_kwargs)
    else:
        visualizer.process_all_om_folders(args.output_dir, args.fps, args.debug_frames, **processing_kwargs)


if __name__ == "__main__":
    main()