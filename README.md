# Truncated-spectrum vs full-spectrum diffusion for graph generation

## Motivation
"Generating graphs via spectral diffusion" (2025) proposes GGSD, which generates graphs by running a diffusion process on the spectral decomposition of the graph Laplacian. Using the whole spectrum makes the diffusion input grow with the number of nodes. This experiment asks whether keeping only a few eigenpairs cuts cost without hurting quality.

## Question
Can a truncated-spectrum diffusion graph generator (k=8 eigenvectors) match the graph-statistics quality of a full-spectrum diffusion generator at a lower training and sampling cost on ENZYMES graphs?

## Dataset
ENZYMES from the public TUDataset collection (https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip), keeping graphs with 20-40 nodes, with a random 80/20 train/test split. This is not the paper's own dataset setup.

## Models
- **Baseline:** a GGSD-style stand-in (the paper's code is not available). It runs a DDPM-style diffusion over the full Laplacian spectrum (all eigenvalues and sign-fixed eigenvectors, padded to 40 nodes). The noise predictor is an sklearn MLPRegressor. Graphs are rebuilt from the Laplacian and thresholded. It is not the paper's method.
- **Improved:** the same pipeline (denoiser size, threshold rule, training budget, evaluation), but only the k=8 smallest eigenpairs are modelled.

Metrics: mean MMD over degree, clustering and Laplacian-spectrum histograms (main, lower is better); wall-clock training plus sampling time and degree-only MMD (secondary). Each model is run with seeds 42, 43 and 44.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> G Minello, A Bicciato, L Rossi (2025). *Generating graphs via spectral diffusion*. International. https://proceedings.iclr.cc/paper_files/paper/2025/hash/b4899ae0d61ed6fc4eadb1a95c5f207f-Abstract-Conference.html

Paper author: Prof. Andrea Torsello, Ca' Foscari University of Venice (andrea.torsello@unive.it)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
