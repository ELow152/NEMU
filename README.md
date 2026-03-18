# NEMU

*NEMU* is my attempt at designing a **full-brain emulator**. NEMU stands for (Neural EMUlation).
It will (eventually) cover the following:

- [ ] A builtin serializer & loader
  - [ ] efficient external streaming of neural and synapse events
     - [ ] safe disk spillover during runtime
     - [ ] adaptive to RAM capacity
  - [ ] Robust handling of imperfect information
     - [ ] error correction on CSV format of data
     - [ ] spacial grid search for synapses

- [ ] A builtin visualizer test
  - [x] Neurons
    - [x] plotted points
  - [ ] Synapses
    - [ ] plotted lines (could be complete)
  - [ ] activations
    - [ ] synapse color gradient change based on activation values
  - [ ] A serializer that actually works
    - [ ] see above

In the future, *NEMU* will cover full-brain emulation, although the specifics are not known.
The biggest bottlenext this far has been serialization and handling of unstructured data.
