# WaveR_TIP

A ViViT-based video classifier where self-attention is replaced by a
**Wave Propagation Operator (WPO)**. Trained on our **LapSurg-230K** dataset
for surgical video action recognition.

This repository is the official implementation of our paper accepted by
**IEEE Transactions on Image Processing (TIP)**:

> **Neural Wave Propagation for Surgical Video Action Recognition:
> A New Dataset and Baseline**

## Install

Install PyTorch matching your CUDA version from <https://pytorch.org>, then:

```bash
pip install -r requirements.txt
```

## Dataset

We introduce **LapSurg-230K**, a large-scale, high-quality and comprehensive
surgical video action recognition dataset. It spans **9 surgery types** and
covers the **11 most fundamental surgical actions**. It is hosted on figshare:

- DOI: <https://doi.org/10.6084/m9.figshare.32237319>

Download and extract it, then place it under `data/LaparoClipsP2/` (this is
the default path expected by the code; you can override it with the
`WAVER_DATA_ROOT` environment variable):

```
data/LaparoClipsP2/
├── clips/
│   ├── 069/
│   │   └── Clip_P069_V001_C1_<Action>_*.mp4
│   ├── 232/
│   ├── 632/
│   ├── 790/
│   └── ...                  # one folder per patient ID
├── train.csv
├── val.csv
└── test.csv
```


`relative_video_path` is relative to `clips/` (so it starts with the patient
ID folder), and `label_name` must be one of the 11 action classes below.

The 11 action classes:

```
AbdominalEntry, Excise, HookCut, Incise, LocPanoView,
NeedleIn, NeedleOut, PanoView, ScissorCut, Suction, UseClip
```

## Train

```bash
python train.py
```

Checkpoints are written to `weights/`, logs to `logs/`.

## Inference

Place a trained `.pkl` checkpoint in `weights/`, then:

```bash
python inference.py
```

The confusion matrix is saved to `output/`.
