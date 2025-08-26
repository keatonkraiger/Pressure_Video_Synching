import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
from scipy.ndimage import median_filter

def load_tekscan_csv(filepath, skip_header_lines=31, rows=60, cols=21):
    frames = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    i = skip_header_lines
    while i < len(lines):
        if lines[i].strip().startswith("Frame"):
            i += 1
            matrix_lines = lines[i:i+rows]
            frame_data = []
            for line in matrix_lines:
                entries = line.strip().split(',')
                entries = entries[:cols] + ['B'] * (cols - len(entries))
                row = [np.nan if val == 'B' else float(val) for val in entries]
                frame_data.append(row)
            frames.append(frame_data)
            i += rows
        else:
            i += 1

    return np.array(frames)  # shape: (N, 60, 21)

def remove_isolated_pixels(frame, min_neighbors=2, kernel_size=3):
    """
    Remove isolated pixels that don't have enough active neighbors.
    
    Args:
        frame: 2D array of pressure values
        min_neighbors: minimum number of non-zero neighbors required
        kernel_size: size of neighborhood to check (3x3, 5x5, etc.)
    """
    # Create binary mask of active pixels (non-zero, non-NaN)
    active_mask = (frame > 0) & (~np.isnan(frame))
    
    # Count active neighbors for each pixel
    kernel = np.ones((kernel_size, kernel_size))
    kernel[kernel_size//2, kernel_size//2] = 0  # Don't count the pixel itself
    neighbor_count = ndimage.convolve(active_mask.astype(int), kernel, mode='constant')
    
    # Keep pixels that have enough neighbors OR are part of a larger structure
    keep_mask = (neighbor_count >= min_neighbors) | (~active_mask)
    
    # Apply the mask
    cleaned_frame = np.copy(frame)
    cleaned_frame[~keep_mask] = 0
    
    return cleaned_frame

def remove_temporal_outliers(data, window_size=5, threshold_std=3):
    """
    Remove temporal outliers by looking at each pixel's behavior over time.
    
    Args:
        data: 3D array (frames, rows, cols)
        window_size: size of temporal window to analyze
        threshold_std: number of standard deviations for outlier detection
    """
    cleaned_data = np.copy(data)
    
    for i in range(data.shape[1]):  # rows
        for j in range(data.shape[2]):  # cols
            pixel_time_series = data[:, i, j]
            
            # Skip if all NaN or all zero
            if np.all(np.isnan(pixel_time_series)) or np.all(pixel_time_series == 0):
                continue
                
            # Use rolling window to detect outliers
            for frame_idx in range(len(pixel_time_series)):
                # Define window around current frame
                start_idx = max(0, frame_idx - window_size // 2)
                end_idx = min(len(pixel_time_series), frame_idx + window_size // 2 + 1)
                window = pixel_time_series[start_idx:end_idx]
                
                # Remove NaN and zeros for statistics
                valid_window = window[(~np.isnan(window)) & (window > 0)]
                
                if len(valid_window) < 3:  # Need at least 3 points for meaningful stats
                    continue
                    
                current_val = pixel_time_series[frame_idx]
                if np.isnan(current_val) or current_val == 0:
                    continue
                    
                # Check if current value is an outlier
                window_mean = np.mean(valid_window)
                window_std = np.std(valid_window)
                
                if window_std > 0 and abs(current_val - window_mean) > threshold_std * window_std:
                    # Additional check: is this value much higher than neighbors?
                    if current_val > window_mean + threshold_std * window_std:
                        cleaned_data[frame_idx, i, j] = 0
    
    return cleaned_data

def apply_median_filter_selective(data, kernel_size=3, high_threshold=500):
    """
    Apply median filter selectively to high-value pixels that might be noise.
    
    Args:
        data: 3D array (frames, rows, cols)
        kernel_size: size of median filter kernel
        high_threshold: threshold above which to apply median filtering
    """
    cleaned_data = np.copy(data)
    
    for frame_idx in range(data.shape[0]):
        frame = data[frame_idx]
        
        # Find pixels above threshold
        high_pixels = frame > high_threshold
        
        if np.any(high_pixels):
            # Apply median filter to the entire frame
            filtered_frame = median_filter(frame, size=kernel_size)
            
            # Only replace the high-value pixels with filtered values
            # But only if the filtered value is significantly lower
            replacement_mask = high_pixels & (filtered_frame < frame * 0.7)
            cleaned_data[frame_idx][replacement_mask] = filtered_frame[replacement_mask]
    
    return cleaned_data

def clip_pressure_values_enhanced(data, min_val, max_val, impossible_threshold=1500):
    """
    Enhanced clipping that handles impossible values more intelligently.
    
    Args:
        data: pressure data array
        min_val: minimum valid pressure
        max_val: maximum valid pressure  
        impossible_threshold: values above this are set to 0 (impossible readings)
    """
    clipped = np.copy(data)
    
    # Set impossible high values to zero instead of max_val
    clipped = np.where(clipped > impossible_threshold, 0, clipped)
    
    # Normal clipping for reasonable range
    clipped = np.where(clipped < min_val, 0, clipped)
    clipped = np.where(clipped > max_val, max_val, clipped)
    
    return clipped

def clean_pressure_data(data, min_val=10, max_val=862, impossible_threshold=1500,
                       remove_isolated=True, remove_temporal=True, apply_median=True):
    """
    Comprehensive pressure data cleaning pipeline.
    
    Args:
        data: 3D pressure data (frames, rows, cols)
        min_val, max_val: normal pressure range
        impossible_threshold: threshold for impossible readings
        remove_isolated: whether to remove isolated pixels
        remove_temporal: whether to remove temporal outliers
        apply_median: whether to apply selective median filtering
    """
    print("  → Starting data cleaning pipeline...")
    
    # Step 1: Enhanced clipping (sets impossible values to 0)
    cleaned = clip_pressure_values_enhanced(data, min_val, max_val, impossible_threshold)
    impossible_count = np.sum(data > impossible_threshold)
    if impossible_count > 0:
        print(f"    • Set {impossible_count} impossible readings (>{impossible_threshold}) to zero")
    
    # Step 2: Remove temporal outliers
    # if remove_temporal:
    #     pre_temporal = np.sum(cleaned > 0)
    #     cleaned = remove_temporal_outliers(cleaned, window_size=7, threshold_std=2.5)
    #     post_temporal = np.sum(cleaned > 0)
    #     temporal_removed = pre_temporal - post_temporal
    #     if temporal_removed > 0:
    #         print(f"    • Removed {temporal_removed} temporal outliers")
    
    # # Step 3: Apply selective median filtering to remaining high values
    # if apply_median:
    #     pre_median = np.copy(cleaned)
    #     cleaned = apply_median_filter_selective(cleaned, kernel_size=3, high_threshold=max_val*0.8)
    #     median_changes = np.sum(pre_median != cleaned)
    #     if median_changes > 0:
    #         print(f"    • Applied median filtering to {median_changes} high-value pixels")
    
    # # Step 4: Remove isolated pixels (do this last to catch any remaining noise)
    # if remove_isolated:
    #     frames_with_isolated = 0
    #     for frame_idx in range(cleaned.shape[0]):
    #         pre_isolated = np.sum(cleaned[frame_idx] > 0)
    #         cleaned[frame_idx] = remove_isolated_pixels(cleaned[frame_idx], 
    #                                                    min_neighbors=1, kernel_size=3)
    #         post_isolated = np.sum(cleaned[frame_idx] > 0)
    #         if post_isolated < pre_isolated:
    #             frames_with_isolated += 1
    #     if frames_with_isolated > 0:
    #         print(f"    • Removed isolated pixels from {frames_with_isolated} frames")
    
    return cleaned

def save_pressure_video(data, output_path, fps=100):
    N = data.shape[0]
    cmap = plt.get_cmap('jet')
    cmap.set_bad(color='black')
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    def combine_frames(frame):
        left = frame[:, :, 0]
        right = frame[:, :, 1]
        combined = np.ma.masked_invalid(np.hstack((left, right)))
        return combined

    initial_frame = combine_frames(data[0])
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(initial_frame, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(10.5, -2, 'Left', ha='center', va='center', fontsize=10)
    ax.text(31.5, -2, 'Right', ha='center', va='center', fontsize=10)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Pressure")

    def update(frame_idx):
        combined = combine_frames(data[frame_idx])
        im.set_data(combined)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=N, blit=True)
    ani.save(output_path, writer='ffmpeg', fps=fps)
    plt.close()
    print(f"  → Video saved to: {output_path}")

from pathlib import Path
def process_all_enhanced(data_dir, save_dir=None, min_val=10, max_val=862, 
                        impossible_threshold=1500, target_fps=50):
    """Enhanced processing with comprehensive cleaning."""
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    files = os.listdir(data_dir)
    processed = []

    for fname in sorted(files):
        if not fname.endswith("_L.csv"):
            continue

        base = fname[:-6]  # Remove _L.csv

        left_path = os.path.join(data_dir, f"{base}_L.csv")
        right_path = os.path.join(data_dir, f"{base}_R.csv")
        
        if not os.path.exists(right_path):
            print(f"[!] Missing right file for: {base}")
            continue

        print(f"[✓] Processing: {base}")
        left = load_tekscan_csv(left_path)
        right = load_tekscan_csv(right_path)

        min_frames = min(len(left), len(right))
        
        # Apply enhanced cleaning to each foot separately
        print("  → Cleaning left foot data...")
        left_cleaned = clean_pressure_data(left[:min_frames], min_val, max_val, impossible_threshold)
        
        print("  → Cleaning right foot data...")
        right_cleaned = clean_pressure_data(right[:min_frames], min_val, max_val, impossible_threshold)
        
        combined = np.stack([left_cleaned, right_cleaned], axis=-1)  # Shape: (N, 60, 21, 2)
        
        # Downsample to the target fps, we assume its recorded at 50 fps.
        combined = combined[::int(100/target_fps)]
        processed.append((base, combined))

        if save_dir:
            save_path = os.path.join(save_dir, f"{base}/Original_Pressure.npy")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, combined)
            print(f"  → Saved to: {save_path}")

            video_path = os.path.join(save_dir, f"{base}/Original_Pressure.mp4")
            print(f"  → Generating video: {video_path}")
            save_pressure_video(combined, video_path, fps=target_fps)

    return processed

# Example usage with enhanced cleaning
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pressure data")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to the save directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir

    # Process with enhanced cleaning
    target_fps = 50
    all_data = process_all_enhanced(data_dir, save_dir,
                                   min_val=10, max_val=862,
                                   impossible_threshold=1500, target_fps=target_fps)