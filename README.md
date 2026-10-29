# AVSnap

This repository contains demo code for the paper <a href = "https://arxiv.org/abs/1808.06250">Dynamic Temporal Alignment of Speech to Lips </a> (Tavi Halperin, Ariel Efrat, and Shmuel Peleg).
The repository reuses code from the demo for the audio-to-video synchronisation network (<a href = "http://www.robots.ox.ac.uk/~vgg/software/lipsync/">SyncNet</a>).

The model and code can be used for research purposes under <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution License</a>.
Please cite the papers below if you make use of the software.

## Prerequisites
- Python >= 3.11
- PyTorch >= 2.15
- ffmpeg and ffprobe (with libx264)

Install with [uv](https://docs.astral.sh/uv/)
```
uv sync
```
or with pip
```
pip install torch
```
The diagnostic plots written by `align_audio.py --save_dp_images` additionally need matplotlib (`uv sync --extra plot`).

## Demo

First, download face detection and SyncNet models
```
sh download_model.sh
```

To run the demo, aligning audio from video2.avi to video from video1.avi
```
python run_pipeline.py data/video1.avi
python run_pipeline.py data/video2.avi
python align_audio.py data/video1.avi data/video2.avi --modality=all
```

The alignment will be computed based on both video and audio of the input videos.
You can use only one of the modalities by changing the flag ```--modality``` to one of ```va/aa/av/vv```, the first letter indicates which modality is used from the first input, and the last letter indicates which is used from the second.

Outputs:
```
data/out/video1_video2/modality_all.avi
data/out/video1_video2/mix.avi
```

### Speed and memory

Memory of `align_audio.py` grows linearly with the input length: the dynamic programming only stores the band of
`MAX_WARP` embeddings around the diagonal, and videos are streamed from ffmpeg instead of loaded whole.

`align_audio.py` runs SyncNet in bfloat16 by default, which only pays off on hardware with bfloat16 support (recent
GPUs, CPUs with AMX or AVX512-BF16). It changes the embeddings by 1-2%; the warp is unchanged on the demo and moves by
at most 8 ms on a 60 s input. `--no-amp` runs it in float32.

## Publications

```
@inproceedings{halperin19dynamic,
  title     =   {Dynamic Temporal Alignment of Speech to Lips‏},
  author    =   {Halperin, Tavi and Ephrat, Ariel and Peleg, Shmuel},
  booktitle =   {2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      =   {2019},
}
```
SyncNet is taken from:
```
@InProceedings{Chung16a,
  author       = "Chung, J.~S. and Zisserman, A.",
  title        = "Out of time: automated lip sync in the wild",
  booktitle    = "Workshop on Multi-view Lip-reading, ACCV",
  year         = "2016",
}
```
