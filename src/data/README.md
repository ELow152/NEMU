# Data handling

This is where Brain data is handled, loaded, and serialized. It will cover the following:

- [ ] Data handling
  - [ ] Serialization
    - [ ] individual neuron information, coordinates, and statistics
    - [ ] individual synapse positions, connections, and statistics
    - [ ] external file streaming to safely buffer objects into memory for serialization
  - [ ] Loading
    - [ ] Load indivudal neuron information, coordinates, and statistics while spilling to disk
    - [ ] Load individual synapse positions, connections, and statistics while spilling to disk
  - [ ] Spilling
    - [ ] Spill to disk safely
      - [ ] Buffer objects and batch computation
      - [ ] Spill to disk during runtime
      - [ ] Serialize on the fly

All three requirements are currently under development. Most work will be dedicated towards serialization in the next push.
