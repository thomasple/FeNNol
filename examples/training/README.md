# Training examples

## Dataset structure
The file `rmd17_aspirin_01.pkl' contains a split of the aspirin dataset from the revMD17 dataset. This split contains 1000 structures for training and 50 structures for validation. Energy units are in kcal/mol and distances in Angstroms.

This dataset is formatted to work with FeNNol's training system. You can inspect it with the following Python code:

```python
import pickle

with open('rmd17_aspirin_01.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.keys()) # dict_keys(['training', 'validation','description'])
print(len(data['training'])) # 1000
print(len(data['validation'])) # 50

print(data['training'][0].keys()) # dict_keys(['species', 'coordinates', 'formation_energy', 'shifted_energy', 'forces'])
```
When preparing your own dataset, make sure to use a similar formatting.

## Training
We provide two example input files: `ani.yaml` and `crate.yaml`.
The first one trains a local model that uses ANI AEVs and a chemical species-specialized neural network to predict energies.
The second one trains a message-passing CRATE model with chemical species embedding and a simple multi-layer perceptron to predict energies.

To train a model, run the following command:
```bash
fennol_train ani.yaml
```
Training on GPU should only take a few minutes. Training on CPU should take about 15min for the ANI model. At the end of training, mean absolute errors on both energy and forces should be below the kcal/mol.

Results will be saved in a directory named `run_dir_aspirin_ani2x` (the `output_directory` key in the input file).
This directory should contain the following files:
- `train.log` contains the training log (a copy of the standard output from the command).
- `config.yaml` is a copy the input file.
- `metrics.traj` contains the evolution of the metrics (for example validation errors) during training. The first line of this file contains the names of the metrics.
- `latest_model.fnx` contains the latest model. This file is saved at each epoch.
- `best_model.fnx` contains the best model. This file is saved each time the validation error is lower than the previous best.
- `final_model.fnx` contains the model at the end of training.

## Run molecular dynamics
After training, copy the `best_model.fnx` to the `examples/md/aspirin/` directory and follow the instructions in the `README.md` file in that directory to run molecular dynamics simulations.

## Examples from the FeNNol paper
The archive file `FeNNol_2405.01491_training_examples.tar.gz` that can be downloaded from Zenodo [https://zenodo.org/records/11146435](https://zenodo.org/records/11146435) contains the output directories for the examples provided in the [FeNNol paper](https://arxiv.org/abs/2405.01491). They contain the model files, training logs, metrics and configuration files. We do not provide the datasets here as they are publicly available on the relevant repositories.