# Truncated-spectrum diffusion graph generation (ENZYMES)

Question: can a diffusion generator that models only the k=8 smallest Laplacian eigenpairs match the graph-statistics quality (mean MMD over degree, clustering and Laplacian-spectrum histograms on held-out test graphs) of a full-spectrum diffusion generator, at lower training and sampling cost?

- Dataset: ENZYMES (TUDataset), graphs with 20-40 nodes; 80/20 train/test split (seed 42); a validation subset of the training graphs is used only to choose the edge threshold.
- Baseline: full-spectrum spectral DDPM with an MLP denoiser (sklearn MLPRegressor).
- Improved: identical pipeline but only the k=8 smallest eigenpairs are modelled.
- Run: `pip install -r requirements.txt && python analysis.py` (CPU only, seeds 42-44). Writes `results.json` and `figures/comparison.png`.

Fix note: `MLPRegressor` is now called with `hidden_layer_sizes=` as a keyword (newer scikit-learn made the first positional argument `loss`), for both models identically.

## Reference

This project is motivated by the following paper:

> G Minello, A Bicciato, L Rossi (2025). *Generating graphs via spectral diffusion*. International. https://proceedings.iclr.cc/paper_files/paper/2025/hash/b4899ae0d61ed6fc4eadb1a95c5f207f-Abstract-Conference.html

Paper author: Prof. Andrea Torsello, Ca' Foscari University of Venice (andrea.torsello@unive.it)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
