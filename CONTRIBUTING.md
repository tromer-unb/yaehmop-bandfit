# Contributing

Contributions are welcome for reproducible scientific input/output, tests and documentation. Please open an issue first for changes to YAeHMOP physics, band indexing, objective definitions or reference provenance.

1. Fork, then create a small topic branch based on main.
2. Set up a virtual environment with `pip install -r requirements.txt`.
3. Run `python -m unittest discover -s tests -p 'test_*.py' -v`. Tests must not install or run DFT, YAeHMOP, optimizers, large k meshes or download large datasets.
4. Add only original, synthetic or explicitly redistributable fixtures. Cite external data/provenance and specify licenses.
5. Document any changed CLI flag, numeric format or error condition. State whether benchmark values are direct fits or frozen-parameter transfer.
6. Open a pull request describing the code changes and the lightweight tests run.

Do not commit solver binaries, third-party atomic tables, third-party DFT band dumps, personal secrets, experiment checkpoints or generated build directories.
