"""
visualize_nmesc.py
==================
Visual diagnostic for NME-SC speaker-count estimation
(Park et al., "Auto-Tuning Spectral Clustering for Speaker Diarization
Using Normalized Maximum Eigengap", IEEE SPL 2019 — arXiv:2003.02405).

Reads the SAME <fid>_embeddings.npz / <fid>_metadata.json pairs your
spectral_clustering.py already consumes, re-runs the NME p-value scan with
every intermediate quantity kept (instead of thrown away), and renders 3
PNGs per recording into its own output sub-folder:

    <output_dir>/<file_id>/01_eigengap_scan.png     g(p) vs p  -> how p* is chosen
    <output_dir>/<file_id>/02_eigenvalue_spectrum.png  sorted eigenvalues at p*,
                                                       max gap highlighted
    <output_dir>/<file_id>/03_eigengap_bar.png       gap size per candidate k,
                                                       winning k highlighted
    <output_dir>/<file_id>/summary.json              numbers behind the plots

Works on ONE file (--file_id fid) or ALL matching pairs found in
--embeddings_dir (default). Each recording gets its own folder, so this is
safe to point at a whole dataset directory in one call.

Usage:
    python visualize_nmesc.py \
        --embeddings_dir ./embeddings_raw \
        --output_dir     ./nmesc_plots \
        --max_speaker    20 \
        --max_rp_threshold 0.25 \
        --sparse_search_volume 30

    # single recording only
    python visualize_nmesc.py --embeddings_dir ./embeddings_raw \
        --output_dir ./nmesc_plots --file_id IS1000a
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh as scipy_eigh
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

BASELINE_DIR = Path(__file__).resolve().parent

# ============================================================================
# Config — edit these, then just run `python vis_num_spk.py` with no flags.
# ============================================================================
EMBEDDINGS_DIR = str(BASELINE_DIR / "v1_output/AMI_mixheadset/embeddings")
OUTPUT_DIR = BASELINE_DIR / "nmesc_plots"
FILE_ID = None             # set to a single recording id (str) to process just that one
MAX_SPEAKER = 25           # matches run_pipeline.py's --max_speaker default
MAX_RP_THRESHOLD = 0.25
SPARSE_SEARCH_VOLUME = 20  # matches cos+sc.py's --sparse_search_volume default
                            # (run_pipeline.py never overrides it)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
})

_scaler = MinMaxScaler(feature_range=(0, 1))


def get_cos_affinity_matrix(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    emb_norm = emb / norms
    sim = cosine_similarity(emb_norm)
    _scaler.fit(sim)
    return _scaler.transform(sim)


def get_k_neighbors_connections(affinity_mat, p_value):
    binarized = np.zeros_like(affinity_mat)
    for i, line in enumerate(affinity_mat):
        idx = np.argsort(line)[::-1][:p_value]
        binarized[idx, i] = 1
    return binarized


def get_affinity_graph_mat(affinity_mat_raw, p_value):
    x = get_k_neighbors_connections(affinity_mat_raw, p_value)
    return 0.5 * (x + x.T)


def is_graph_fully_connected(affinity_mat):
    return get_largest_component(affinity_mat, 0).sum() == affinity_mat.shape[0]


def get_largest_component(affinity_mat, seg_index):
    n = affinity_mat.shape[0]
    connected = np.zeros(n, dtype=bool)
    frontier = np.zeros(n, dtype=bool)
    frontier[seg_index] = True
    for _ in range(n):
        prev = connected.sum()
        np.logical_or(connected, frontier, out=connected)
        if prev >= connected.sum():
            break
        idxs = np.where(frontier)[0]
        frontier.fill(False)
        for i in idxs:
            np.logical_or(frontier, affinity_mat[i].astype(bool), out=frontier)
    return connected


def get_laplacian(x):
    x = x.copy()
    x[np.diag_indices(x.shape[0])] = 0
    d = np.diag(np.sum(np.abs(x), axis=1))
    return d - x


def eig_decompose(laplacian):
    return scipy_eigh(laplacian)


def get_lambda_gap_list(lambdas):
    lambdas = np.real(lambdas)
    return list(lambdas[1:] - lambdas[:-1])


def get_p_value_list(n, max_rp_threshold, sparse_search_volume):
    max_n = int(n * max_rp_threshold)
    max_n = max(max_n, 2)
    n_search = min(max_n, sparse_search_volume)
    return list(np.linspace(1, max_n, n_search, endpoint=True).astype(int)), max_n


def nme_scan(mat, max_speaker, max_rp_threshold, sparse_search_volume, eps=1e-10):
    """
    Full instrumented p-value scan. Returns a dict with everything needed
    to plot the eigengap-estimation process (nothing here is discarded the
    way NMESC.NMEanalysis() discards it).
    """
    n = mat.shape[0]
    p_value_list, max_n = get_p_value_list(n, max_rp_threshold, sparse_search_volume)

    per_p = []  # one record per candidate p_value
    for p in p_value_list:
        affinity = get_affinity_graph_mat(mat, p)
        laplacian = get_laplacian(affinity)
        lambdas, _ = eig_decompose(laplacian)
        lambdas = np.sort(np.real(lambdas))
        gap_list = get_lambda_gap_list(lambdas)
        k_range = gap_list[: min(max_speaker, len(gap_list))]
        est_k = int(np.argmax(k_range)) + 1
        max_gap = k_range[est_k - 1]
        g_p = (p / n) / (max_gap / (max(lambdas) + eps) + eps)
        per_p.append({
            "p_value": int(p),
            "est_num_spk": est_k,
            "g_p": float(g_p),
            "lambdas": lambdas,
            "gap_list": np.array(gap_list),
            "fully_connected": bool(is_graph_fully_connected(affinity)),
        })

    g_p_values = np.array([r["g_p"] for r in per_p])
    best_idx = int(np.argmin(g_p_values))
    best = per_p[best_idx]

    # raw affinity graph actually used downstream at the chosen p*
    return {
        "n_segments": n,
        "p_value_list": p_value_list,
        "max_n": max_n,
        "per_p": per_p,
        "best_idx": best_idx,
        "best_p_value": best["p_value"],
        "best_g_p": best["g_p"],
        "estimated_num_speakers": best["est_num_spk"],
        "best_lambdas": best["lambdas"],
        "best_gap_list": best["gap_list"],
    }


# ============================================================================
# Plots — each one shows a single step of "how NME-SC finds k" visually.
# ============================================================================

def plot_eigengap_scan(result, file_id, out_dir, max_speaker):
    """Step 1: sweep p_value (starting at p=1, the real production sparse
    search points from nme_scan) -> the NME ratio r(p); the minimum of this
    curve is p-hat, the neighborhood size NME-SC commits to."""
    ps = [r["p_value"] for r in result["per_p"]]
    gs = [r["g_p"] for r in result["per_p"]]
    ks = [r["est_num_spk"] for r in result["per_p"]]
    p_hat = result["best_p_value"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ps, gs, "-o", color="#3B6EA5", markersize=4, linewidth=1.5, zorder=2)
    ax.scatter([p_hat], [result["best_g_p"]],
               s=180, facecolor="#E4572E", edgecolor="black", zorder=3,
               label=r"$\hat{p}$" + f" = {p_hat}  (min r(p))")
    ax.axvline(p_hat, color="#E4572E", linestyle="--", alpha=0.5)

    ax.set_xlim(min(ps), max(ps))
    ax.set_yscale("log")

    for p, g, k in zip(ps, gs, ks):
        if p == p_hat:
            ax.annotate(f"k={k}", (p, g), textcoords="offset points",
                        xytext=(8, 10), fontsize=10, fontweight="bold", color="#E4572E")

    ax.set_xlabel("p-neighbors (kNN sparsification level)")
    ax.set_ylabel("r(p)")
    ax.set_title(r"NME scan — " + file_id + "\nsmallest r(p) picks the pruning level $\\hat{p}$")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "01_eigengap_scan.png", dpi=150)
    plt.close(fig)


def plot_eigenvalue_spectrum(result, file_id, out_dir):
    """Step 2: the sorted Laplacian eigenvalues AT p*, with the single
    largest consecutive gap (among the first max_speaker candidates)
    highlighted — that gap position IS the speaker-count estimate."""
    lambdas = result["best_lambdas"]
    gap_list = result["best_gap_list"]
    k = result["estimated_num_speakers"]
    show_n = min(len(lambdas), max(20, k + 8))

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(show_n)
    ax.plot(xs, lambdas[:show_n], "-o", color="#444444", markersize=4, zorder=2)

    # shade + arrow across the winning gap (between eigenvalue k-1 and k, 0-indexed)
    gap_lo, gap_hi = k - 1, k
    ax.axvspan(gap_lo, gap_hi, color="#E4572E", alpha=0.15, zorder=1)
    ax.annotate(
        "", xy=(gap_hi, lambdas[gap_hi]), xytext=(gap_lo, lambdas[gap_lo]),
        arrowprops=dict(arrowstyle="<->", color="#E4572E", lw=2),
    )
    mid_y = (lambdas[gap_lo] + lambdas[gap_hi]) / 2
    ax.text(gap_hi + 0.3, mid_y, f"largest gap\n(gap idx {gap_lo}\u2192{gap_hi})\n"
            f"\u2192 estimated k = {k}",
            color="#E4572E", fontsize=10, fontweight="bold", va="center")

    ax.set_xlabel("eigenvalue index (sorted ascending)")
    ax.set_ylabel("Laplacian eigenvalue \u03bb")
    ax.set_title(r"Eigenvalue spectrum at $\hat{p}$ = " + f"{result['best_p_value']} — {file_id}\n"
                f"the biggest jump = the boundary between speaker clusters")
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "02_eigenvalue_spectrum.png", dpi=150)
    plt.close(fig)


def plot_eigengap_bar(result, file_id, out_dir, max_speaker):
    """Step 3: eigengap size for every candidate speaker count k=1..max_speaker,
    at p*. argmax of these bars = estimated_num_speakers."""
    gap_list = result["best_gap_list"]
    show_n = min(max_speaker, len(gap_list))
    ks = np.arange(1, show_n + 1)
    gaps = gap_list[:show_n]
    winner = result["estimated_num_speakers"]

    colors = ["#E4572E" if kk == winner else "#3B6EA5" for kk in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ks, gaps, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(ks)
    ax.set_xlabel("candidate number of speakers k  (\u03bb_k \u2212 \u03bb_{k-1})")
    ax.set_ylabel("eigengap magnitude")
    ax.set_title(r"Eigengap per candidate k at $\hat{p}$ = " + f"{result['best_p_value']} — {file_id}\n"
                f"tallest bar wins \u2192 estimated k = {winner}")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "03_eigengap_bar.png", dpi=150)
    plt.close(fig)


# ============================================================================
# I/O + driver
# ============================================================================

def load_recording(npz_path):
    data = np.load(npz_path)
    if isinstance(data, np.ndarray):
        return data
    keys = list(data.keys())
    return data[keys[0]]


def process_one(npz_path, output_dir, max_speaker, max_rp_threshold, sparse_search_volume):
    file_id = npz_path.stem.replace("_embeddings", "")
    emb = load_recording(npz_path)
    n = emb.shape[0]
    if n < 4:
        print(f"[skip] {file_id}: only {n} segments, too few for eigengap scan")
        return None

    mat = get_cos_affinity_matrix(emb)
    result = nme_scan(mat, max_speaker=max_speaker,
                      max_rp_threshold=max_rp_threshold,
                      sparse_search_volume=sparse_search_volume)

    rec_dir = output_dir / file_id
    rec_dir.mkdir(parents=True, exist_ok=True)

    plot_eigengap_scan(result, file_id, rec_dir, max_speaker)
    plot_eigenvalue_spectrum(result, file_id, rec_dir)
    plot_eigengap_bar(result, file_id, rec_dir, max_speaker)

    summary = {
        "file_id": file_id,
        "n_segments": n,
        "best_p_value": result["best_p_value"],
        "best_g_p": result["best_g_p"],
        "estimated_num_speakers": result["estimated_num_speakers"],
        "p_value_search_range": [int(result["p_value_list"][0]), int(result["p_value_list"][-1])],
        "max_n_considered": result["max_n"],
    }
    with open(rec_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] {file_id}: N={n:4d}  p*={result['best_p_value']:3d}  "
          f"est_k={result['estimated_num_speakers']:2d}  -> {rec_dir}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Visualize NME-SC eigengap speaker-count estimation")
    p.add_argument("--embeddings_dir", default=str(EMBEDDINGS_DIR),
                   help="Directory with *_embeddings.npz files (metadata json not required here). "
                        f"Default: {EMBEDDINGS_DIR}")
    p.add_argument("--output_dir", default=str(OUTPUT_DIR),
                   help="Root output dir; each recording gets its own <file_id>/ sub-folder. "
                        f"Default: {OUTPUT_DIR}")
    p.add_argument("--file_id", default=FILE_ID,
                   help="Process only this recording (optional).")
    p.add_argument("--max_speaker", type=int, default=MAX_SPEAKER,
                   help="Max candidate speaker count to search for the eigengap bar/spectrum plots.")
    p.add_argument("--max_rp_threshold", type=float, default=MAX_RP_THRESHOLD,
                   help="Max ratio of segments used as the p-value search ceiling (same as NMESC).")
    p.add_argument("--sparse_search_volume", type=int, default=SPARSE_SEARCH_VOLUME,
                   help="Number of p-values sampled across the search range.")
    return p.parse_args()


def main():
    args = parse_args()
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(embeddings_dir.glob("*_embeddings.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No *_embeddings.npz files found in {embeddings_dir}")

    if args.file_id:
        npz_files = [f for f in npz_files if f.stem.replace("_embeddings", "") == args.file_id]
        if not npz_files:
            raise FileNotFoundError(f"No embeddings file found for file_id={args.file_id}")

    summaries = []
    for npz_path in npz_files:
        s = process_one(npz_path, output_dir, args.max_speaker,
                        args.max_rp_threshold, args.sparse_search_volume)
        if s:
            summaries.append(s)

    with open(output_dir / "all_summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nDone. {len(summaries)} recording(s) plotted under {output_dir}/<file_id>/")


if __name__ == "__main__":
    main()