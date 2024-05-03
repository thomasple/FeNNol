## FeNNol: Force-field-enhanced Neural Networks optimized library
FeNNol is a library for building, training and running neural network potentials for molecular simulations. It is based on the JAX library and is designed to be fast and flexible.

FeNNol's documentation is available [here](https://thomasple.github.io/FeNNol/).

### Warning
FeNNol is in early stages of development: some APIs are subject to change and documentation is still minimal (but should improve soon !). 
This is also a temporary repository for the preprint https://arxiv.org/abs/2405.01491 The library will soon be moved to a more permanent location.

## Installation
You can start with a fresh environment using conda:
```bash
conda create -n fennol python=3.10 pip
conda activate fennol
```

The first step is to install jax (see details at: https://jax.readthedocs.io/en/latest/installation.html).
For a conda installation with CUDA 11.8, use:
```bash
conda install jaxlib=*=*cuda11* jax=0.4.25 cuda-nvcc=11.8 -c conda-forge -c nvidia
```
and make sure that jax uses the correct version of CUDA/CuDNN/ptxas by correctly setting your PATH and LD_LIBRARY_PATH.
For example, you can update the `LD_LIBRARY_PATH` as following so that jax uses the CUDA libraries that you just installed with conda:
```bash
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/fennol/lib:$LD_LIBRARY_PATH
```

Then, you can clone and install FeNNol using pip:
```bash
git clone ...
cd FeNNol
pip install .
```

*Optional dependencies*:
- Some modules require e3nn-jax (https://github.com/e3nn/e3nn-jax) which can be installed with:
```bash
pip install --upgrade e3nn-jax
```
- The provided training script requires pytorch (at least the cpu version) for dataloaders:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
- For the Deep-HP interface, cffi, pydlpack and pycuda are required:
```bash
conda install cffi pydlpack pycuda
```
