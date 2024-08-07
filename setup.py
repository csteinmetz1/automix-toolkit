# from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name="automix-toolkit",
    version="0.0.1",
    description="Models and datasets for training deep learning automatic mixing models",
    author="Christian J. Steinmetz",
    author_email="c.j.steinmetz@qmul.ac.uk",
    url="https://github.com/csteinmetz1/automix",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "pytorch_lightning",
        "tqdm",
        "numpy==1.26.0",
        "matplotlib",
        "pedalboard",
        "scipy",
        "auraloss",
        "wget",
        "pyloudnorm",
        "scikit-learn",
        "librosa",
        "soundfile",
        "tensorboard",
    ],
)
