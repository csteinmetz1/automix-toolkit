<div align="center">

# automix-toolkit
Models and datasets for training deep learning automatic mixing models

</div>

# Setup

```
python -m venv env 
source env/bin/activate
pip install --upgrade pip
```

```
pip install -e . 
```

Eventually we want `pip install automix`

Download the demo dataset.

Right now we could build a demo dataset using some creative commons mixes
http://c4dm.eecs.qmul.ac.uk/multitrack/MixEvaluation/
Or we could build our own dataset using loops from Logic

# Usage

## 1. Pretrained models

## 2. Training

### Datasets 

| Name        | Mixes | Size (GB) | Download | 
|-------------|-------|-----------|----------|
| ENST-Drums  |       |           |          |
| MedleyDB    | 
| 

# Planning 

- [x] Add wandb logging
- [x] Use sum and difference loss 
- [] Working WaveUNet
- [] Working DMC
- [x] Datasets 
    - [] MedleyDB
    - [x] ENST-Drums
    - [] Mixing Secrets
    - [] Data augmentation 

We may want to check for silence when loading the mix.

We may want to add some data augmentation. 
- Randomize the gain of the stems during mixing
