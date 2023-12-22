<div align="center">

<img width="80px" src="docs/assets/logo.png"> 

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
git clone https://github.com/csteinmetz1/automix-toolkit.git
cd automix-toolkit
pip install -e . 
```

# Usage

## 1. Pretrained models

First you need to download the pretrained models into the [`automix/checkpoints/`](automix/checkpoints/) directory. 

```
wget https://huggingface.co/csteinmetz1/automix-toolkit/resolve/main/enst-drums-dmc.ckpt
wget https://huggingface.co/csteinmetz1/automix-toolkit/resolve/main/enst-drums-mixwaveunet.ckpt
mv enst-drums-dmc.ckpt checkpoints/enst-drums-dmc.ckpt
mv enst-drums-mixwaveunet.ckpt checkpoints/enst-drums-mixwaveunet.ckpt
```

You can run inference with a model checkpoint and a directory of stems.
If the pretrained model expects a certain track ordering ensure that you the 
tracks in the provided directory are labeled `01_...`, `02_...`, etc.
For demonstration, we can download a drum multitrack from the test set. 

```
wget https://huggingface.co/csteinmetz1/automix-toolkit/resolve/main/drums-test-rock.zip
unzip drums-test-rock.zip
```

Then we can generate a mix by calling the script and passing the checkpoint and directory of tracks.
This will create an output in the current directory called `mix.wav`.

```
python scripts/inference.py checkpoints/enst-drums-dmc.ckpt drums-test-rock/tracks
```

## 2. Training

### Datasets 

| Name         | Mixes | Size (GB)    | Download | 
|--------------|-------|--------------|----------|
| ENST-Drums   | 210   | 20 GB        | [link](https://perso.telecom-paristech.fr/grichard/ENST-drums/) |
| MedleyDB     | 197   | 82 + 71 GB   | [link](https://medleydb.weebly.com/) |
| DSD100       | 100   | 14 GB        | [link](http://liutkus.net/DSD100.zip) |
| DSD100subset |   4   | 0.1 GB       | [link](https://www.loria.fr/~aliutkus/DSD100subset.zip)

### Configurations

We provide training recipes to reproduce the pretrained models across a range of architectures and datasets. 
You can find shell scripts for each configuration. Note that you will need to update the paths to reflect your local file system after downloading the datasets. 

```
./configs/drums_dmc.sh
./configs/drums_mixwaveunet.sh
./configs/drums_wet_mixwaveunet.sh

./dsd100_dmc.sh

./medleydb_dmc.sh
./medleydb_mixwaveunet.sh
```

# Notebooks

We also provide interactive notebooks to demonstrate the functionality of this toolkit. 
To use the notebooks first ensure you have installed the `automix` package. We suggest using conda or another virtual environemnt system. After installing `automix` package, then install jupyter.
Then you can launch the notebooks.

```
python -m venv env 
source env/bin/activate
pip install -e .
pip install jupyter
jupyter notebook notebooks/
```

- [Inference](notebooks/01_inference.ipynb) - In this notebook we demonstrate how to download and use pretrained models to create multitrack mixes of drum recordings. 
- [Datasets](notebooks/02_datasets.ipynb) - In this notebook we provide an overview of supplied datasets.
- [Models](notebooks/03_models.ipynb) - In this notebook you can explore the Mix-Wave-U-Net and Differentiable Mixing Console models
- [Training](notebooks/04_training.ipynb) - In this notebook you can train your own model on the ENST-drums dataset. 
- [Evaluation](notebooks/05_evaluate.ipynb) - In this notebook you can evaluate mixes via objective metrics.