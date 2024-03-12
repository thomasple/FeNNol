# FeNNol
### Force-field-enhanced Neural Networks optimized library
FeNNol is a library for building, training and running neural network potentials for molecular simulations. It is based on the JAX library and is designed to be fast and flexible.


## Installation
You can start with a fresh environment using conda:
```bash
conda create -n fennol python=3.10 pip
conda activate fennol
```

The first step is to install jax (see details at: https://jax.readthedocs.io/en/latest/installation.html).
For a conda installation with CUDA 11.8, use:
```bash
conda install jaxlib=*=*cuda11* jax cuda-nvcc=11.8 -c conda-forge -c nvidia
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
pip install -e .
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



## TODO:
- [X] Preprocessing system 
- [ ] Improve neighbor lists (PBC, parallelization, GPU, external calls...)
- [X] Add e3 tensor products 
      - [X] start by filtered TPs
      - [X] and then maybe general FullyConnectedTP -> do we need e3nn-jax ?
      - [X] If needed, add custom e3Linear layer
- [X] Port modules from torchnff
- [X] Convert ANI models parameters
- [X] Add a MD code (with adQTB) -> test efficiency with ANI
      - [X] classical MD
      - [X] QTB
      - [X] RPMD
      - [ ] spectra
      - [ ] histograms
- [X] Implement a save/load system like in torchnff
- [X] Add method to load a pretrained model
- [X] generic training system ?
      - [ ] more control over the optimizer
      - [ ] more flexible data loading (conversion script ?)
- [ ] Add tests (?)
- [ ] Add examples
- [ ] DOCUMENTATION / TUTORIALS