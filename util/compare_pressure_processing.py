import subprocess
from glob import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import argparse
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import median_filter
import cv2


def _get_video_info(video_path: Path):
    """Get video information (fps and frame count) using ffprobe."""
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

def _ffmpeg_extract(video_path: Path, out_dir: Path, fps: int = None):
    """Extract frames from video to out_dir/%06d.jpg if not already present.
       If fps is provided, resample the video to that fps.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # if frames already exist, skip
    if any(out_dir.glob('*.jpg')):
        return
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video_path)]
    if fps:  # enforce fps resampling
        cmd.extend(['-vf', f'fps={fps}'])
    cmd.append(str(out_dir / '%06d.jpg'))
    subprocess.run(cmd, check=True)

def _load_cached_frames(frame_dir: Path):
    files = sorted(glob(str(frame_dir / '*.jpg')))
    return files  # return file paths, we'll lazy-read per frame

def remove_isolated_pixels(frame, min_neighbors=2, kernel_size=3):
    """Remove isolated pixels that don't have enough active neighbors."""
    active_mask = (frame > 0) & (~np.isnan(frame))
    kernel = np.ones((kernel_size, kernel_size))
    kernel[kernel_size//2, kernel_size//2] = 0
    neighbor_count = ndimage.convolve(active_mask.astype(int), kernel, mode='constant')
    keep_mask = (neighbor_count >= min_neighbors) | (~active_mask)
    cleaned_frame = np.copy(frame)
    cleaned_frame[~keep_mask] = 0
    return cleaned_frame

def remove_temporal_outliers(data, window_size=5, threshold_std=3):
    """Remove temporal outliers by looking at each pixel's behavior over time."""
    cleaned_data = np.copy(data)
    
    for i in range(data.shape[1]):  # rows
        for j in range(data.shape[2]):  # cols
            pixel_time_series = data[:, i, j]
            
            if np.all(np.isnan(pixel_time_series)) or np.all(pixel_time_series == 0):
                continue
                
            for frame_idx in range(len(pixel_time_series)):
                start_idx = max(0, frame_idx - window_size // 2)
                end_idx = min(len(pixel_time_series), frame_idx + window_size // 2 + 1)
                window = pixel_time_series[start_idx:end_idx]
                
                valid_window = window[(~np.isnan(window)) & (window > 0)]
                
                if len(valid_window) < 3:
                    continue
                    
                current_val = pixel_time_series[frame_idx]
                if np.isnan(current_val) or current_val == 0:
                    continue
                    
                window_mean = np.mean(valid_window)
                window_std = np.std(valid_window)
                
                if window_std > 0 and abs(current_val - window_mean) > threshold_std * window_std:
                    if current_val > window_mean + threshold_std * window_std:
                        cleaned_data[frame_idx, i, j] = 0
    
    return cleaned_data

def apply_median_filter_selective(data, kernel_size=3, high_threshold=500):
    """Apply median filter selectively to high-value pixels that might be noise."""
    cleaned_data = np.copy(data)
    
    for frame_idx in range(data.shape[0]):
        frame = data[frame_idx]
        high_pixels = frame > high_threshold
        
        if np.any(high_pixels):
            filtered_frame = median_filter(frame, size=kernel_size)
            replacement_mask = high_pixels & (filtered_frame < frame * 0.7)
            cleaned_data[frame_idx][replacement_mask] = filtered_frame[replacement_mask]
    
    return cleaned_data

def clip_pressure_values_enhanced(data, min_val, max_val, impossible_threshold=1500):
    """Enhanced clipping that handles impossible values more intelligently."""
    clipped = np.copy(data)
    clipped = np.where(clipped > impossible_threshold, 0, clipped)
    clipped = np.where(clipped < min_val, 0, clipped)
    clipped = np.where(clipped > max_val, max_val, clipped)
    return clipped

def clean_pressure_data(data, min_val=10, max_val=862, impossible_threshold=1500,
                       remove_isolated=True, remove_temporal=True, apply_median=True):
    """Comprehensive pressure data cleaning pipeline."""
    print("  → Starting data cleaning pipeline...")
    
    cleaned = clip_pressure_values_enhanced(data, min_val, max_val, impossible_threshold)
    impossible_count = np.sum(data > impossible_threshold)
    if impossible_count > 0:
        print(f"    • Set {impossible_count} impossible readings (>{impossible_threshold}) to zero")
    
    if remove_temporal:
        pre_temporal = np.sum(cleaned > 0)
        cleaned = remove_temporal_outliers(cleaned, window_size=7, threshold_std=2.5)
        post_temporal = np.sum(cleaned > 0)
        temporal_removed = pre_temporal - post_temporal
        if temporal_removed > 0:
            print(f"    • Removed {temporal_removed} temporal outliers")
    
    if apply_median:
        pre_median = np.copy(cleaned)
        cleaned = apply_median_filter_selective(cleaned, kernel_size=3, high_threshold=max_val*0.8)
        median_changes = np.sum(pre_median != cleaned)
        if median_changes > 0:
            print(f"    • Applied median filtering to {median_changes} high-value pixels")
    
    if remove_isolated:
        frames_with_isolated = 0
        for frame_idx in range(cleaned.shape[0]):
            pre_isolated = np.sum(cleaned[frame_idx] > 0)
            cleaned[frame_idx] = remove_isolated_pixels(cleaned[frame_idx], 
                                                       min_neighbors=1, kernel_size=3)
            post_isolated = np.sum(cleaned[frame_idx] > 0)
            if post_isolated < pre_isolated:
                frames_with_isolated += 1
        if frames_with_isolated > 0:
            print(f"    • Removed isolated pixels from {frames_with_isolated} frames")
    
    return cleaned

def preprocess_pressure_block(pressure_block, mode: str):
    """Process pressure data and return processed data with value range."""
    # unify to (T, H, W) with left|right concatenated for visualization
    if pressure_block.ndim == 4 and pressure_block.shape[-1] == 2:
        left = pressure_block[..., 0]
        right = pressure_block[..., 1]
        data = np.concatenate([left, right], axis=2)  # (T,H,2W)
    elif pressure_block.ndim == 3:
        data = pressure_block
    else:
        raise ValueError(f"Unexpected pressure data shape: {pressure_block.shape}")

    if mode == 'standard':
        clipped = np.clip(data, 0, 862)
        vmin, vmax = 0, 862
        return clipped, (vmin, vmax)

    if mode == 'percentile':
        clipped = np.clip(data, 0, 1500)
        valid = clipped[np.isfinite(clipped)]
        if valid.size == 0:
            return np.clip(data, 0, 862), (0, 862)
        p95 = max(200, np.percentile(valid, 95))
        clipped = np.clip(clipped, 0, p95)
        return clipped, (0, p95)

    if mode == 'all':
        T, H, W2 = data.shape
        W = W2 // 2
        left = data[..., :W]
        right = data[..., W:]
        left_clean = clean_pressure_data(left, min_val=10, max_val=862, impossible_threshold=1500,
                                         remove_isolated=True, remove_temporal=True, apply_median=True)
        right_clean = clean_pressure_data(right, min_val=10, max_val=862, impossible_threshold=1500,
                                          remove_isolated=True, remove_temporal=True, apply_median=True)
        cleaned = np.concatenate([left_clean, right_clean], axis=2)
        valid = cleaned[np.isfinite(cleaned) & (cleaned > 0)]
        if valid.size == 0:
            return np.clip(cleaned, 0, 862), (0, 862)
        p95 = max(200, np.percentile(valid, 95))
        cleaned = np.clip(cleaned, 0, p95)
        return cleaned, (0, p95)

    raise ValueError(f"Unknown pressure mode: {mode}")


class DatasetVisualizer:
    def __init__(self, root_dir, person_name):
        self.root_dir = Path(root_dir)
        self.person_name = person_name
        
        # OM task mapping
        self.om_titles = {
            'OM1': 'Circular Walking',
            'OM2': 'Straight Walking', 
            'OM3': 'Lateral Walking',
            'OM4': 'Single Leg Stance',
            'OM5': 'Calf Raise',
            'OM6': 'Squat',
            'OM7': 'Lunges',
            'OM8': 'Leg Kick',
            'OM9': 'Push and Pull',
            'OM10': 'Throwing',
            'OM11': 'Warm Up'
        }
    
    def load_config(self, om_folder):
        """Load configuration JSON file"""
        # Use wildcard of path and file ending with json
        json_files = list(om_folder.glob("*.json"))
        if json_files:
            with open(json_files[0], 'r') as f:
                return json.load(f)
        return {"offset": 0}  # Default if no config

    def load_pressure_data(self, om_folder):
        """Load foot pressure numpy data"""
        npy_files = list(om_folder.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy file found in {om_folder}")
        
        pressure_file = npy_files[0]
        data = np.load(pressure_file)
        return data

    def create_visualization(self, om_folder, output_path=None, cache_frames=False, fps=50,
                            modes=('standard',), debug_frames=None):
        """Create visualization with matched panel heights and preserved video aspect ratios."""
        print(f"Processing {om_folder.name}...")
        # --- target fps: downsample to the minimum ---
        target_fps = fps
        
        # paths
        v1_path = om_folder / f"Video_V1.mp4"
        v2_path = om_folder / f"Video_V2.mp4"
        if not v1_path.exists() or not v2_path.exists():
            print(f"Video files not found for {om_folder.name}")
            return

        # load pressure data
        pressure_data = self.load_pressure_data(om_folder)
        
        # Get the number of frames in V1, V2, and pressure
        # Run ffprobe to get fps and frame count of V1 and V2
        v1_fps, v1_frame_count = _get_video_info(v1_path)
        v2_fps, v2_frame_count = _get_video_info(v2_path)
        pressure_frame_count = pressure_data.shape[0] if pressure_data is not None else 0

        assert v1_frame_count == v2_frame_count == pressure_frame_count, \
            f"Frame counts do not match: V1={v1_frame_count}, V2={v2_frame_count}, Pressure={pressure_frame_count}"
        assert v1_fps == v2_fps, f"V1 and V2 fps do not match: V1={v1_fps}, V2={v2_fps}"
        assert v1_fps == target_fps, f"Input video fps ({v1_fps}) does not match target fps ({target_fps})"

        # preprocess pressure for all modes
        pressure_processed = {}
        pressure_ranges = {}
        
        for mode in modes:
            processed, (vmin, vmax) = preprocess_pressure_block(pressure_data, mode)
            pressure_processed[mode] = processed
            pressure_ranges[mode] = (vmin, vmax)

        # cache dirs for videos
        output_parent_dir = output_path.parent
        om_idx = om_folder.name
        cache_root = output_parent_dir / "_cache" / om_idx
        v1_dir = cache_root / "V1_frames"
        v2_dir = cache_root / "V2_frames"

        if cache_frames:
            _ffmpeg_extract(v1_path, v1_dir, fps=target_fps)
            _ffmpeg_extract(v2_path, v2_dir, fps=target_fps)

        v1_files = _load_cached_frames(v1_dir) if cache_frames else None
        v2_files = _load_cached_frames(v2_dir) if cache_frames else None

        # synchronize
        start_cut = 0
        # determine video dimensions
        if v1_files:
            sample_frame = plt.imread(v1_files[start_cut])
        else:
            cap1 = cv2.VideoCapture(str(v1_path))
            cap1.set(cv2.CAP_PROP_POS_FRAMES, start_cut)
            ok1, fr1 = cap1.read()
            cap1.release()
            sample_frame = cv2.cvtColor(fr1, cv2.COLOR_BGR2RGB) if ok1 else np.zeros((480, 640, 3))
        video_h, video_w = sample_frame.shape[:2]
        video_aspect = video_w / video_h

        # pressure aspect ratio
        p_h, p_w = pressure_processed[modes[0]].shape[1:3]
        pressure_aspect = p_w / p_h

        # GridSpec width ratios: based on aspect ratios only
        width_ratios = [video_aspect, video_aspect] + [pressure_aspect] * len(modes)
        num_panels = 2 + len(modes)

        # Figure setup
        fig = plt.figure(figsize=(12, 5), dpi=100)
        fig.patch.set_facecolor('white')

        gs = GridSpec(
            1, num_panels, figure=fig,
            width_ratios=width_ratios,
            wspace=0.2, left=0.02, right=0.95, top=0.90, bottom=0.05
        )
        ims = []
        title_size = 12

        # --- View 1 ---
        ax = fig.add_subplot(gs[0, 0])
        ax.set_title("View 1", fontsize=title_size, pad=4)
        ax.axis("off")
        v1_first = plt.imread(v1_files[start_cut]) if v1_files else sample_frame
        im1 = ax.imshow(v1_first, aspect='equal')
        ims.append(im1)

        # --- View 2 ---
        ax = fig.add_subplot(gs[0, 1])
        ax.set_title("View 2", fontsize=title_size, pad=4)
        ax.axis("off")
        if v2_files:
            v2_first = plt.imread(v2_files[start_cut])
        else:
            v2_first = np.zeros_like(sample_frame)
        im2 = ax.imshow(v2_first, aspect='equal')
        ims.append(im2)

        # --- Pressure panels ---
        for i, mode in enumerate(modes):
            ax = fig.add_subplot(gs[0, 2+i])
            if mode == 'standard':
                ax.set_title("Raw\nPressure", fontsize=title_size,pad=2)
            else:
                ax.set_title(f"Pressure\n{mode}", fontsize=title_size, pad=2)  # stacked title
            ax.axis("off")  # hide axes so they don’t overlap V2
            
            vmin, vmax = pressure_ranges[mode]
            pressure_frame = pressure_processed[mode][start_cut]
            im = ax.imshow(pressure_frame, cmap="jet", vmin=vmin, vmax=vmax, aspect='equal')

            ticks = np.linspace(vmin, vmax, 5)
            cbar = plt.colorbar(
                im, ax=ax,
                fraction=0.05,   # allocate more space for ticks/label
                pad=0.05,
                ticks=ticks
            )
            cbar.ax.tick_params(labelsize=title_size-5)

            ims.append(im)

        # --- sync length ---
        if v1_files:
            v1_count = len(v1_files)
            v2_count = len(v2_files)
        else:
            v1_cap_probe = cv2.VideoCapture(str(v1_path))
            v2_cap_probe = cv2.VideoCapture(str(v2_path))
            v1_count = int(v1_cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
            v2_count = int(v2_cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
            v1_cap_probe.release(); v2_cap_probe.release()
        T = pressure_data.shape[0]
        sync_len = min(v1_count - start_cut, v2_count - start_cut, T - start_cut)
        if debug_frames is not None:
            sync_len = min(sync_len, debug_frames)
            print(f"Debug mode: limiting to {sync_len} frames")

        if sync_len <= 0:
            print("No overlapping frames after offset correction.")
            return

        # --- animation function ---
        def animate(frame_num):
            actual_frame = start_cut + frame_num
            if actual_frame >= sync_len + start_cut:
                return ims

            # v1
            if v1_files and actual_frame < len(v1_files):
                ims[0].set_array(plt.imread(v1_files[actual_frame]))

            # v2
            if v2_files and actual_frame < len(v2_files):
                ims[1].set_array(plt.imread(v2_files[actual_frame]))

            # pressure
            for j, mode in enumerate(modes):
                if actual_frame < pressure_processed[mode].shape[0]:
                    ims[2+j].set_array(pressure_processed[mode][actual_frame])

            return ims

        # --- save ---
        out_mp4 = output_path if output_path else self.root_dir / f"{om_folder.name}_visualization.mp4"
        writer = animation.FFMpegWriter(
            fps=target_fps, bitrate=8000,
            extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
        )
        anim = animation.FuncAnimation(fig, animate, frames=sync_len,
                                    interval=1000/target_fps, blit=True, repeat=False)
        anim.save(str(out_mp4), writer=writer, dpi=120)
        plt.close(fig)
        print(f"[OK] {out_mp4} with modes: {', '.join(modes)}")
        
    def process_all_om_folders(self, output_dir=None, cache_frames=False, fps=30,
                            modes=('standard',), debug_frames=None):
        """Process all OM folders under root_dir/person_name."""
        person_root = self.root_dir / self.person_name

        if output_dir is None:
            output_dir = self.root_dir / "visualizations" / self.person_name
        else:
            output_dir = Path(output_dir) / self.person_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all OM folders in root_dir/person_name
        om_folders = [f for f in person_root.iterdir()
                    if f.is_dir() and f.name.startswith("OM")]
        if not om_folders:
            print(f"[WARN] No OM folders found in {self.root_dir}")
            return
        
        print(f"[INFO] Found {len(om_folders)} OM folders")
        for om_folder in sorted(om_folders):
            print(f"[INFO] Processing {om_folder.name}")
            out_path = output_dir / f"{om_folder.name}_visualization.mp4"
            try:
                self.create_visualization(
                    om_folder,
                    output_path=out_path,
                    cache_frames=cache_frames,
                    fps=fps,
                    modes=modes,
                    debug_frames=debug_frames
                )
            except Exception as e:
                print(f"[ERROR] Failed on {om_folder.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Create visualizations for multi-modal dataset')
    parser.add_argument('--root_dir', help='Root directory containing OM folders')
    parser.add_argument('--person_name', help='Person\'s name for the title')
    parser.add_argument('--output_dir', help='Output directory for visualizations')
    parser.add_argument('--om_folder', help='Process specific OM folder only')
    parser.add_argument('--cache_frames', action='store_true',
                        help='Extract and reuse per-modality frames with ffmpeg.')
    parser.add_argument('--fps', type=int, default=50)
    parser.add_argument('--debug_frames', type=int, default=None,
                        help='Limit processing to first N frames for debugging')
    parser.add_argument('--pressure_mode', nargs='+',
                        choices=['standard', 'percentile', 'all'],
                        default=['standard'],
                        help="One or more modes: standard (clip only), "
                            "percentile (95th vmax), all (full cleaning pipeline).")
    
    args = parser.parse_args()
    
    visualizer = DatasetVisualizer(args.root_dir, args.person_name)
    
    if args.om_folder:
        print(f"[INFO] Processing single OM folder: {args.om_folder}")
        om_path = Path(args.root_dir) / args.om_folder
        if not om_path.exists():
            print(f"OM folder not found: {om_path}")
            return
        
        output_path = None
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{args.om_folder}_visualization.mp4"
        
        visualizer.create_visualization(
            om_path,
            output_path,
            cache_frames=args.cache_frames,
            fps=args.fps,
            modes=args.pressure_mode,
            debug_frames=args.debug_frames
        )
    else:
        print("[INFO] Processing all OM folders")
        visualizer.process_all_om_folders(
            output_dir=args.output_dir,
            cache_frames=args.cache_frames,
            fps=args.fps,
            modes=args.pressure_mode,
            debug_frames=args.debug_frames
        )

if __name__ == "__main__":
    main()