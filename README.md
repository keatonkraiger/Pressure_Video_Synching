# Data Synchronization

## Installation

You should first install the required packages:

```
pip install -r reqs.txt
```

You may also need to install FFMPEG (`conda install -c conda-forge ffmpeg`)

## Preparation

We recommend starting with two directories. First is the main data directory. For OM data, this should look something like: 

```
Data
  |-Alice
  |  |-OM1
  |  |  |-OM1_V1.mp4
  |  |  |-OM1_V2.mp4
  |  |-OM2
  |  |  |-OM2_V1.mp4
  |  |  |-OM2_V2.mp4
  |  |-...
  |  |  |-...
  ```

Then a folder with the raw pressure data formatted something like:
```
Raw_pressure
  |-Alice
  |  |-OM1_L.csv
  |  |-OM1_R.csv
  |  |-OM2_L.csv
  |  |-OM2_R.csv
  |  |-...
```
In the above data, I have one subject with N OM takes.

## Pressure Processing

To first processing the raw pressure run the command

```
python clean_pressure.py --data_dir Raw_pressure --save_dir Data
```

This will generate files like this for each OM dir:
```
Data
  |-Alice
  |  |-OM1
  |  |  |-OM1_V1.mp4
  |  |  |-OM1_V2.mp4
  |  |  |-OM1_Original_Pressure.npy
  |  |  |-OM1_Original_Pressure.mp4
```

**Important**: The above code assumes the foot pressure is recorded at 100fps. If it is not, you'll need to adjust this manually.

## Data Synchronization

To then synchronize the data, you'll run 

```
python syncher.py
```

You should then load in one of the video views by clicking the `Load RGB Video` button (OM1_V1.mp4 for example), then the pressure video with the `Load Pressure Video` (Original_Pressure.mp4 for example).

**Important**: The synching can only be done if BOTH videos are at 50 fps. The pressure cleaning script subsamples the pressure to 50fps (make sure its originally 100 fps). If the RGB video is not at 50 fps the syncher will give you an option to run ffmpeg and convert your video. This requires it to be installed (`conda install -c conda-forge ffmpeg`). If it doesn't work in the syncher you could run the command from your terminal: 

```
ffmpeg -i input.mp4 -r 50 output.mp4
```

**Important**: After synching it, click the `Save Sync Config` button and save the configuration file in the appropriate directory and name it `config.json`. For example, save the config file for OM1 as `Data/Alice/OM1/config.json`.

This will be done for each OM folder resulting in a directory structure like this:

```
Data
  |-Alice
  |  |-OM1
  |  |  |-OM1_V1.mp4
  |  |  |-OM1_V2.mp4
  |  |  |-OM1_Original_Pressure.npy
  |  |  |-OM1_Original_Pressure.mp4
  |  |  |-config.json
```

## Final Dataset Creation

Once you've processed the foot pressure, synched and saved the config you may run the following script to generate clean files for each OM:

```
python create_final_data.py --input_path Data/Alice --save_dir Cleaned/Alice
```

To run a single OM, you may pass `--input_path` a specific OM directory, for example:
```
python create_final_data.py --input_path Data/Alice/OM1 --save_dir Cleaned/Alice
```
