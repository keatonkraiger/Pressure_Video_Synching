# Creating OM Dataset

## Pipeline Overview
This pipeline processes multimodal motion capture data (video, pressure, OpenPose, mocap) and synchronizes everything to 50fps.

## Steps

### 1. Convert videos to 50fps
Standardize all videos to 50fps for consistent processing.
```bash
python scripts/convert_all_fps.py --root_dir raw_data/Video/<subject_name>
```

**Important**: Track which videos had their fps changed (e.g., 25fps→50fps). You'll need this information in step 3.

---

### 2. Clean and process pressure data
Generate cleaned pressure `.npy` files and videos at 50fps.
```bash
python scripts/clean_pressure.py \
    --data_dir raw_data/Pressure_CSVs/<subject_name> \
    --save_dir untrimmed/Pressure/<subject_name>
```

---

### 3. Generate synchronization offsets
Manually determine frame offsets between pressure and video data using the syncher tool.
```bash
python syncher.py
```

For each video, create a JSON config file in your offsets directory: `synch_offsets/<subject_name>/OM<idx>.json`

**Critical**: If a video's fps was changed in step 1, add `"original_video_fps"` to the config:
```json
{
  "offset": -50,
  "original_video_fps": 25.0
}
```

If the video was already 50fps, you can omit `original_video_fps` (defaults to 50.0).

---

### 4. Run OpenPose
Process each 50fps video through OpenPose to extract 2D keypoints. Output should go to `raw_data/openpose_output/<subject_name>/OM<idx>/OM<idx>_V[1|2]_jsons/`.

---

### 5. Convert OpenPose JSONs to NPY format
Convert OpenPose JSON outputs to consolidated `.npy` arrays.
```bash
python scripts/format_OP_jsons.py --name <subject_name>
```

Optional arguments:
- `--specific_oms` (e.g., `1 3` to process only OM1 and OM3)

---

### 6. Triangulate 3D poses
Generate 3D poses from stereo 2D OpenPose keypoints.
```bash
python scripts/triangulate_oms.py --subjects <subject_name>
```

Optional arguments:
- `--specific_oms` (e.g., `OM1 OM3`)
- `--animate` (generate animation videos)
- `--dry_run` (don't save files)

**Important**: For subjects Keaton, Kyle, Varad, Spencer, and Abhinav, do NOT use `--no_world_fix` flag (world-frame correction is required).

---

### 7. Extract Vicon model outputs
Extract CoM, joints, and markers from Vicon JSON files.
```bash
python scripts/get_model_output.py --person <subject_name>
```

**Important**: For subjects Keaton, Kyle, Varad, Spencer, and Abhinav, do NOT use `--no_world_fix` flag (Rz(-90°) transformation is required).

---

### 8. Combine and synchronize all modalities
Combine all data streams (video, pressure, OpenPose, mocap) with proper synchronization.
```bash
python scripts/combine_all.py \
    --name <subject_name> \
    --offsets_dir synch_offsets
```

Optional arguments:
- `--specific_oms` (e.g., `1 3` to process only OM1 and OM3)

---

## Post-processing Pressure Cleaning

Some of the takes have faulty pressure sensors. Fixing them is a trial and error process.


## Directory Structure
```
raw_data/
├── Video/<subject_name>/OM<idx>_V[1|2].mp4
├── Pressure_CSVs/<subject_name>/OM<idx>_[L|R].csv
├── openpose_output/<subject_name>/OM<idx>/OM<idx>_V[1|2]_jsons/
├── XCPs/<subject_name>.xcp
└── Model_output/<subject_name>/OM<idx>.json

untrimmed/
├── Pressure/<subject_name>/OM<idx>/Original_Pressure.npy
├── BODY25/<subject_name>/OM<idx>/BODY25_V[1|2].npy, BODY25_3D.npy
└── Model_output/<subject_name>/OM<idx>/MOCAP_3D.npy, CoM.npy, etc.

synch_offsets/
└── <subject_name>/OM<idx>.json

Complete/
└── <subject_name>/OM<idx>/
    ├── Video_V[1|2].mp4
    ├── pressure.npy
    ├── BODY25_V[1|2].npy, BODY25_3D.npy
    ├── MOCAP_3D.npy, MOCAP_MRK.npy
    └── CoM.npy, CoM_floor.npy
```

---

## Notes

- All final data in `Complete/` is synchronized at 50fps
- The pipeline intelligently handles different mocap sampling rates (25Hz, 50Hz, 100Hz)
- Vicon data is automatically resampled to match video/OpenPose frame counts
- Frame offsets account for recording start time differences between pressure and video systems