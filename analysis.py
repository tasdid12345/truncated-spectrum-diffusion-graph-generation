"""Truncated (k=8) vs full-spectrum spectral diffusion graph generators on ENZYMES."""
import io, json, os, sys, time, warnings, zipfile
import numpy as np, requests
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
URL = "https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip"
NMAX, T, HID, EPOCHS, REP, NGEN, SEEDS = 40, 50, (256, 256), 40, 40, 100, [42, 43, 44]
NAMES = ["degree", "clustering", "spectrum"]

def download(url):  # retry with backoff on 429/5xx
    for i in range(5):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200: return r.content
            print("HTTP", r.status_code)
        except requests.RequestException as e: print(e)
        time.sleep(2 ** i)
    raise RuntimeError("download failed")

def load_graphs():  # adjacency matrices of ENZYMES graphs with 20-40 nodes
    z = zipfile.ZipFile(io.BytesIO(download(URL)))
    rd = lambda s: io.StringIO(z.read(f"ENZYMES/ENZYMES_{s}.txt").decode())
    ind = np.loadtxt(rd("graph_indicator"), dtype=int) - 1
    ed = np.loadtxt(rd("A"), delimiter=",", dtype=int) - 1
    gi = ind[ed[:, 0]]
    graphs = []
    for g in range(ind.max() + 1):
        nodes = np.where(ind == g)[0]; n = len(nodes)
        if not 20 <= n <= 40: continue
        e = ed[gi == g] - nodes[0]
        A = np.zeros((n, n)); A[e[:, 0], e[:, 1]] = 1
        A = np.maximum(A, A.T); np.fill_diagonal(A, 0); graphs.append(A)
    return graphs

# ---------- evaluation (shared) ----------
def stats(A):  # normalised histograms: degree, clustering coeff, normalised-Laplacian spectrum
    d = A.sum(1); pairs = d * (d - 1) / 2
    c = np.divide(np.diag(A @ A @ A) / 2, pairs, out=np.zeros_like(d), where=pairs > 0)
    dn = np.sqrt(np.maximum(d, 1))
    ev = np.linalg.eigvalsh(np.eye(len(A)) - A / np.outer(dn, dn))
    h = [np.bincount(d.astype(int), minlength=NMAX)[:NMAX],
         np.histogram(c, bins=10, range=(0, 1))[0], np.histogram(ev, bins=20, range=(0, 2))[0]]
    return [x / x.sum() for x in h]

def ref_stats(gs): return [np.array(h) for h in zip(*map(stats, gs))]

def mmd(X, Y, sigma=0.1):  # biased MMD^2, Gaussian kernel on total-variation distance
    k = lambda a, b: np.exp(-(0.5 * cdist(a, b, "cityblock")) ** 2 / (2 * sigma ** 2)).mean()
    return k(X, X) + k(Y, Y) - 2 * k(X, Y)

def score(gs, ref): return np.array([mmd(a, b) for a, b in zip(ref_stats(gs), ref)])

# ---------- spectral representation ----------
def featurize(A, K):  # K smallest eigenpairs of L=D-A, sign-fixed, zero-padded to NMAX nodes
    n = len(A); lam, U = np.linalg.eigh(np.diag(A.sum(1)) - A)
    U = U * np.sign(U[np.abs(U).argmax(0), np.arange(n)])
    m = min(n, K); Up = np.zeros((NMAX, K)); Up[:n, :m] = U[:, :m]
    lp = np.zeros(K); lp[:m] = lam[:m]
    return np.concatenate([lp, Up.ravel()])

def decode(x, n, K):  # rebuild Laplacian, return edge scores S=-L_offdiag
    lam = np.clip(x[:K], 0, None); m = min(n, K)
    U = x[K:].reshape(NMAX, K)[:n, :m]
    S = -(U * lam[:m]) @ U.T; np.fill_diagonal(S, -np.inf)
    return S

def run(K, seed, tr, val_ref, test_ref):
    rng = np.random.default_rng(seed); t0 = time.time()
    X = np.array([featurize(A, K) for A in tr]); mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-6] = 1; X = (X - mu) / sd
    beta = np.linspace(1e-3, 0.2, T); al = 1 - beta; ab = np.cumprod(al)
    idx = np.repeat(np.arange(len(X)), REP); t = rng.integers(0, T, len(idx))
    eps = rng.standard_normal((len(idx), X.shape[1]))
    xt = np.sqrt(ab[t])[:, None] * X[idx] + np.sqrt(1 - ab[t])[:, None] * eps
    net = MLPRegressor(HID, batch_size=128, max_iter=EPOCHS, random_state=seed).fit(np.c_[xt, t / T], eps)
    x = rng.standard_normal((NGEN, X.shape[1]))  # ancestral DDPM sampling
    for s in reversed(range(T)):
        e = net.predict(np.c_[x, np.full(NGEN, s / T)])
        x = (x - beta[s] / np.sqrt(1 - ab[s]) * e) / np.sqrt(al[s])
        if s > 0: x = x + np.sqrt(beta[s]) * rng.standard_normal(x.shape)
        x = np.clip(x, -5, 5)
    sizes = rng.choice([len(A) for A in tr], NGEN)  # sizes drawn from training sizes
    S = [decode(xi, n, K) for xi, n in zip(x * sd + mu, sizes)]
    cost = time.time() - t0
    # threshold rule: pooled-score quantile grid, chosen on validation graphs (from train only)
    pooled = np.concatenate([s[np.triu_indices(len(s), 1)] for s in S])
    best = min(np.quantile(pooled, np.linspace(0.5, 0.99, 25)),
               key=lambda th: score([(s > th).astype(float) for s in S], val_ref).mean())
    sc = score([(s > best).astype(float) for s in S], test_ref)
    return dict(mmd=sc.mean(), deg=sc[0], clus=sc[1], spec=sc[2], time=cost, dim=X.shape[1])

if __name__ == "__main__":
    graphs = load_graphs(); print("graphs kept:", len(graphs))
    trall, te = train_test_split(graphs, test_size=0.2, random_state=42)
    tr, va = train_test_split(trall, test_size=0.2, random_state=42)  # val only for threshold choice
    val_ref, test_ref = ref_stats(va), ref_stats(te)
    res = {"baseline": [], "improved": []}
    for seed in SEEDS:
        for name, K in [("baseline", NMAX), ("improved", 8)]:
            r = run(K, seed, tr, val_ref, test_ref); res[name].append(r)
            print(f"seed {seed} {name}: MMD={r['mmd']:.4f} deg={r['deg']:.4f} time={r['time']:.1f}s", flush=True)
    agg = lambda m, key: float(np.mean([r[key] for r in res[m]]))
    sd = lambda m, key: float(np.std([r[key] for r in res[m]]))
    metrics = {}
    for m in res:
        for key, nm in [("mmd", "score"), ("deg", "deg_mmd"), ("clus", "clus_mmd"), ("spec", "spec_mmd"), ("time", "time_s")]:
            metrics[f"{m}_{nm}"] = agg(m, key); metrics[f"{m}_{nm}_std"] = sd(m, key)
        metrics[f"{m}_input_dim"] = float(res[m][0]["dim"])
    metrics["improvement"] = metrics["baseline_score"] - metrics["improved_score"]
    metrics["speedup_x"] = metrics["baseline_time_s"] / metrics["improved_time_s"]
    if not all(np.isfinite(v) for v in metrics.values()): sys.exit("non-finite metrics")
    out = {
        "question": "Can a truncated-spectrum diffusion graph generator (k=8 eigenvectors) match the graph-statistics quality of a full-spectrum diffusion generator at a lower training and sampling cost on ENZYMES graphs?",
        "method": "Spectral DDPM (T=50, sklearn MLPRegressor noise predictor) on sign-fixed Laplacian eigenpairs padded to 40 nodes; Laplacian rebuilt and thresholded (threshold chosen on a validation split of train graphs). Full spectrum vs k=8 smallest eigenpairs; 3 model seeds; MMD vs test graphs.",
        "dataset": {"name": "ENZYMES (TUDataset), graphs with 20-40 nodes", "source_url": URL, "n_rows": len(graphs),
                    "n_train": len(trall), "n_test": len(te), "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "Mean MMD over degree, clustering and Laplacian-spectrum histograms (test graphs)",
        "higher_is_better": False,
        "baseline": "GGSD-style stand-in: diffusion over the full Laplacian spectrum (all eigenvalues/eigenvectors, padded to 40) with an MLP denoiser.",
        "improved": "Same diffusion, denoiser size, threshold rule, budget and evaluation, but only the k=8 smallest eigenpairs are modelled (~5x smaller input).",
        "metrics": metrics,
        "notes": "Not the paper's code or dataset; MLP denoiser stands in for the paper's neural network. Biased MMD^2 with Gaussian-TV kernel (sigma=0.1), 100 generated graphs per run, small test set so MMD is noisy. Time = featurisation + training + sampling (CPU, wall-clock, includes shared-machine noise). Rebuilding L from only the 8 smallest eigenpairs is a low-rank approximation that discards high-frequency structure; the threshold is a single global value chosen on validation graphs from train, never on test. Eigenvector sign ambiguity is only partly fixed (degenerate eigenspaces remain ambiguous).",
    }
    with open("results.json", "w", encoding="utf-8") as f: json.dump(out, f, indent=2)
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    for a, (nm, lab) in zip(ax, [("score", "Mean MMD (lower better)"), ("time_s", "Train+sample time (s)")]):
        a.bar(["full", "k=8"], [metrics[f"baseline_{nm}"], metrics[f"improved_{nm}"]],
              yerr=[metrics[f"baseline_{nm}_std"], metrics[f"improved_{nm}_std"]], color=["gray", "tab:blue"])
        a.set_ylabel(lab)
    plt.tight_layout(); plt.savefig("figures/comparison.png", dpi=120)
    print(json.dumps(metrics, indent=2))
