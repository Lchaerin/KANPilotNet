# _KANPilotNet_: Advanced PilotNet with KAN


# How to Use
The instructions are tested on Ubuntu 18.04 with python 3.8 and tensorflow 2.10.0 (CUDA 11.2 and cuDNN 8.1).

## Installation
- Clone the PilotNet repository:
   ```bash
   $ git clone https://github.com/Lchaerin/KANPilotNet.git
   ```

- Create a conda environment
   ```bash
   $ conda create -n [env_name] python=3.8
   $ conda activate [env_name]
   ```

## Dataset
### Option 1 (Small)
　If you want to run the demo on the dataset or try some training works, download the
[driving_dataset.zip](https://drive.google.com/file/d/1Ue4XohCOV5YXy57S_5tDfCVqzLr101M7/view) and recommend you to
extract into the dataset folder [`./data/datasets/`](./data/datasets/).
```bash
$ cd $ROOT/data/datasets/
$ wget -t https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view?usp=sharing
$ unzip driving_dataset.zip -d .
```

　This [driving_dataset.zip](https://drive.google.com/file/d/1Ue4XohCOV5YXy57S_5tDfCVqzLr101M7/view) consists of
**images of the road ahead (`*.jpg`)** and recorded **steering wheel angles (`%.6f`)**, `data.txt` should in following
format:
```yaml
    ...
98.jpg 2.120000
99.jpg 2.120000
100.jpg 2.120000
101.jpg 2.120000
    ...
```
　**PS:** The Official download link [driving_dataset.zip](https://drive.google.com/file/d/1Ue4XohCOV5YXy57S_5tDfCVqzLr101M7/view) is on Google Drive, here I also share a backup link in Baidu Net: [download link](https://pan.baidu.com/s/1kZC-6CL1xgk2SUtCt2oz5A) (extract code: **gprm**).

### Option 2 (Big)
※ This dataset is not fully checked. Option 1 is recommended.

Download a chunk(or chunks) from [comma2k19 dataset](https://github.com/commaai/comma2k19).

Then, run `export_frames_with_angles.py`
```bash
$ python export_frames_with_angles.py ~/[root]/KANPilotNet/Chunk_1 ~/[root]/KANPilotNet/data/datasets/driving_dataset --width 455 --height 256 --jpeg-quality 70
```

## Demo
　You can run this demo directly on a live webcam feed in actual running scenario (**online**) or just **offline**, given input
images of the road ahead.

+ Either rename the file you want to use among `pilotNet_original.py` (original (PilotNet)[https://github.com/AutoDeep/PilotNet]), `pilotNet_KAN1.py` (CNN+DenseKAN), and `PilotNet_KAN2.py` (ConvKAN+DenseKAN) located in `./src/nets` to `pilotNet.py`; or edit the import statements in `run_capture.py`, `run_dataset.py`, and `train.py` in `./src` from 
   ```python
   from nets.pilotNet import PilotNet
   ```
to ```python
   from nets.[desired model architecture] import PilotNet
   ```


+ Run the model on the dataset.
   ```bash
   $ ./scripts/demo.sh
   ```
+ Run the model on a live webcam feed
   ```bash
   $ ./scripts/demo.sh -online
   ```

## Training/Validation
+ After downloading the dataset, you can train your own model parameters as following:
   ```bash
   $ ./scripts/train.sh
   ```
   + You can run `./scripts/train.sh` to train your model from downloaded dataset following tips above. Training logs and 
   model will be stored into [./logs](./logs) and [./logs/checkpoint](./logs/checkpoint) respectively.
   + `-dataset_dir` can help you to specify other available dataset.
   + You can use `-log_dir` to set another log directory, and be careful to use `-f` for log files synchronization,
   fix `WARNING:tensorflow:Found more than one metagraph event per run. Overwriting the metagraph with the newest event.`
   + You can use `-num_epochs` and `-batch_size` to control the training step if good at it.


