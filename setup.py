from distutils.core import setup

setup(
    name="automix-toolkit",
    version="0.0.1",
    description="Models and datasets for training deep learning automatic mixing models",
    author="Christian J. Steinmetz",
    author_email="c.j.steinmetz@qmul.ac.uk",
    url="https://github.com/csteinmetz1/automix",
    packages=["automix"],
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "pytorch_lightning",
        "tqdm",
        "numpy",
        "matplotlib",
        "pedalboard",
        "scipy",
        "auraloss",
        "wget",
        "pyloudnorm",
        "sklearn"
    ],
)
