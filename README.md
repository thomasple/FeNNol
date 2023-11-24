# FeNNol
Force-field-enhanced Neural Networks optimized library


## TODO:
- [X] Preprocessing system 
- [ ] Improve neighbor lists (PBC, parallelization, GPU, external calls...)
- [X] Add e3 tensor products 
      - [X] start by filtered TPs
      - [X] and then maybe general FullyConnectedTP -> do we need e3nn-jax ?
      - [X] If needed, add custom e3Linear layer
- [ ] Port modules from torchnff
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
- [ ] DOCUMENTATION / TUTORIALS