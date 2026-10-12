# Truncated-spectrum diffusion graph generation (ENZYMES)

Question: can a diffusion generator that models only the k=8 smallest Laplacian eigenpairs match the graph-statistics quality (mean MMD over degree, clustering and Laplacian-spectrum histograms on held-out test graphs) of a full-spectrum diffusion generator, at lower training and sampling cost?

- Dataset: ENZYMES (TUDataset), graphs with 20-40 nodes; 80/20 train/test split (seed 42); a validation subset of the training graphs is used only to choose the edge threshold.
- Baseline: full-spectrum spectral DDPM with an MLP denoiser (sklearn MLPRegressor).
- Improved: identical pipeline but only the k=8 smallest eigenpairs are modelled.
- Run: `pip install -r requirements.txt && python analysis.py` (CPU only, seeds 42-44). Writes `results.json` and `figures/comparison.png`.

Fix note: `MLPRegressor` is now called with `hidden_layer_sizes=` as a keyword (newer scikit-learn made the first positional argument `loss`), for both models identically.

## Results (latest run)

On ENZYMES, the k=8 truncated-spectrum generator scored 0.141 mean MMD (lower is better) versus 0.159 for a full-spectrum GGSD-style MLP-denoiser baseline on the same test graphs, a small gain at about 3.8x lower runtime, with spectrum-histogram MMD slightly worse.

| Metric | Value |
|---|---|
| baseline_score | 0.1589 |
| baseline_score_std | 0.0008241 |
| baseline_deg_mmd | 0.1314 |
| baseline_deg_mmd_std | 0.0001628 |
| baseline_clus_mmd | 0.1774 |
| baseline_clus_mmd_std | 0.01188 |
| baseline_spec_mmd | 0.1679 |
| baseline_spec_mmd_std | 0.01383 |
| baseline_time_s | 51.37 |
| baseline_time_s_std | 1.69 |
| baseline_input_dim | 1640 |
| improved_score | 0.1415 |
| improved_score_std | 0.003216 |
| improved_deg_mmd | 0.1315 |
| improved_deg_mmd_std | 0.0003034 |
| improved_clus_mmd | 0.1208 |
| improved_clus_mmd_std | 0.01947 |
| improved_spec_mmd | 0.1722 |
| improved_spec_mmd_std | 0.0108 |
| improved_time_s | 13.45 |
| improved_time_s_std | 0.09931 |
| improved_input_dim | 328 |
| improvement | 0.01744 |
| speedup_x | 3.821 |

Run on 2026-10-12. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Not the paper's code or dataset; MLP denoiser stands in for the paper's neural network. Biased MMD^2 with Gaussian-TV kernel (sigma=0.1), 100 generated graphs per run, small test set so MMD is noisy. Time = featurisation + training + sampling (CPU, wall-clock, includes shared-machine noise). Rebuilding L from only the 8 smallest eigenpairs is a low-rank approximation that discards high-frequency structure; the threshold is a single global value chosen on validation graphs from train, never on test. Eigenvector sign ambiguity is only partly fixed (degenerate eigenspaces remain ambiguous).

## Reference

This project is motivated by the following paper:

> G Minello, A Bicciato, L Rossi (2025). *Generating graphs via spectral diffusion*. International. https://proceedings.iclr.cc/paper_files/paper/2025/hash/b4899ae0d61ed6fc4eadb1a95c5f207f-Abstract-Conference.html

Paper author: Prof. Andrea Torsello, Ca' Foscari University of Venice (andrea.torsello@unive.it)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
