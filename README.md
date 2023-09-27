# FeNNol
Force-field-enhanced Neural Networks optimized library


TODO:
- [X] Preprocessing system 
- [ ] Improve neighbor lists (PBC, parallelization, GPU, external calls...)
- [ ] Add e3 tensor products 
      - [ ] start by filtered TPs
      - [ ] and then maybe general FullyConnectedTP -> do we need e3nn-jax ?
      - [ ] If needed, add custom e3Linear layer
- [ ] Port modules from torchnff
- [ ] Convert ANI models parameters
- [ ] Add a MD code (with adQTB) -> test efficiency with ANI
- [X] Implement a save/load system like in torchnff
- [ ] generic training system ?
- [ ] Add tests (?)
- [ ] DOCUMENTATION / TUTORIALS