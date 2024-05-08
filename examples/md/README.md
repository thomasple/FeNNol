# Molecular Dynamics Examples
This directory contains examples for running molecular dynamics simulations with FeNNol models. We provide the `ani2x.fnx` model file which stores the first model of the ANI-2x ensemble, with the weights adapted from torchANI. 

You can navigate to any repository and run the following command to start a simulation:
```bash
fennol_md input.fnl
```
The file `input.fnl` contains the full configuration for the molecular dynamics simulation.

List of examples:
- `aspirin`: gas-phase aspirin (no periodic boundary conditions).
- `watertiny`: 27 water molecules in a periodic box (used to check PBC simulations without minimum image convention)
- `watersmall`,`waterbig`,`waterbox`,`waterhuge`: increasing number of water molecules in a periodic box (used to check performance with PBC and minimum image convention)
- `dhfr`: solvated DHFR protein (around 20K atoms)

### computing radial distribution functions
The `watersmall/compute_rdf.py` and `watertiny/compute_rdf_fullpbc.py` scripts can be used to compute radial distributions from a `.arc` trajectory file (an modified XYZ file format used by Tinker to save trajectories). For example, to compute the RDF for the `watersmall` example, run the following command after running (or during) the simulation in the `watersmall` directory:
```bash
python compute_rdf.py watersmall.arc -w -t 10
```
This will produce a `gr.dat` file with the partial radial distribution functions.
The `-w` flag says that we are watching changes in the file (so it will update the rdf as the simulation progresses) and the `-t` flag specifies the number of frames to skip (thermalization). For `watersmall` the `gr.dat` file can be visually compared to the provided `gr.dat.ref` that was obtain from a 1 ns ANI-2x classical Langevin simulation (with timestep 0.5 fs).

This script can be easily adapted for other examples by changing the `cell`, `rmax`, `dr` and `pairs` variables in the `main` function.