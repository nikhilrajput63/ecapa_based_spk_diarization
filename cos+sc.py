"""
Auto-Tuning Spectral Clustering for Speaker Diarization
========================================================
Adapted from:
  - tango4j/Auto-Tuning-Spectral-Clustering (MIT License)
  - NVIDIA NeMo nmesc_clustering.py (Apache 2.0)

Reference:
  Park et al., "Auto-Tuning Spectral Clustering for Speaker Diarization
  Using Normalized Maximum Eigengap", IEEE SPL, 2019.
  https://arxiv.org/abs/2003.02405

Input format (per recording):
  <file_id>_embeddings.npz   – numpy array, shape (N, D), NOT L2-normalised
  <file_id>_metadata.json    – list of dicts with keys:
                               segment_id, file_id, start, end

Usage:
  python spectral_clustering.py \\
      --embeddings_dir ./embeddings_raw \\
      --output_dir     ./clustering_output \\
      --max_speaker    8 \\
      --spt_est_thres  NMESC \\
      --threshold      None \\
      --reco2num_spk   None
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from sklearn.cluster._kmeans import k_means
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ── optional torch eigen decomposition ──────────────────────────────────────
try:
    import torch
    from torch.linalg import eigh as torch_eigh
    TORCH_EIGH = True
except ImportError:
    TORCH_EIGH = False

scaler = MinMaxScaler(feature_range=(0, 1))

# ============================================================================
# Graph / affinity helpers
# ============================================================================

def isGraphFullyConnected(affinity_mat):
    return getTheLargestComponent(affinity_mat, 0).sum() == affinity_mat.shape[0]


def getTheLargestComponent(affinity_mat, seg_index):
    """BFS to find the connected component containing seg_index."""
    num_of_segments = affinity_mat.shape[0]
    connected_nodes  = np.zeros(num_of_segments, dtype=bool)
    nodes_to_explore = np.zeros(num_of_segments, dtype=bool)
    nodes_to_explore[seg_index] = True

    for _ in range(num_of_segments):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            neighbors = affinity_mat[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)

    return connected_nodes


def getKneighborsConnections(affinity_mat, p_value):
    """Binarize top-p values for each row."""
    binarized = np.zeros_like(affinity_mat)
    for i, line in enumerate(affinity_mat):
        sorted_idx = np.argsort(line)[::-1]
        indices    = sorted_idx[:p_value]
        binarized[indices, i] = 1
    return binarized


def getAffinityGraphMat(affinity_mat_raw, p_value):
    """Binarize and symmetrize the affinity matrix."""
    X = getKneighborsConnections(affinity_mat_raw, p_value)
    return 0.5 * (X + X.T)


def getMinimumConnection(mat, max_N, n_list):
    """Add edges until the graph is fully connected."""
    p_value      = 1
    affinity_mat = getAffinityGraphMat(mat, p_value)
    for p_value in n_list:
        fully_connected = isGraphFullyConnected(affinity_mat)
        affinity_mat    = getAffinityGraphMat(mat, p_value)
        if fully_connected or p_value > max_N:
            break
    return affinity_mat, p_value

# ============================================================================
# Cosine affinity matrix
# ============================================================================

def getCosAffinityMatrix(emb):
    """
    L2-normalise embeddings, compute cosine similarity, then
    min-max scale to [0, 1].
    """
    # L2 normalise (embeddings are NOT pre-normalised)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    emb_norm = emb / norms

    sim_d = cosine_similarity(emb_norm)
    scaler.fit(sim_d)
    sim_d = scaler.transform(sim_d)
    return sim_d

# ============================================================================
# Laplacian & eigen decomposition
# ============================================================================

def getLaplacian(X):
    X = X.copy()
    X[np.diag_indices(X.shape[0])] = 0
    D = np.diag(np.sum(np.abs(X), axis=1))
    return D - X


def eigDecompose(laplacian, cuda=False, device=None):
    if TORCH_EIGH:
        if cuda:
            device = device or torch.cuda.current_device()
            lap_t  = torch.from_numpy(laplacian).float().to(device)
        else:
            lap_t  = torch.from_numpy(laplacian).float()
        lambdas, diffusion_map = torch_eigh(lap_t)
        return lambdas.cpu().numpy(), diffusion_map.cpu().numpy()
    else:
        return scipy_eigh(laplacian)


def getLambdaGapList(lambdas):
    lambdas = np.real(lambdas)
    return list(lambdas[1:] - lambdas[:-1])


def estimateNumofSpeakers(affinity_mat, max_num_speaker, is_cuda=False):
    laplacian        = getLaplacian(affinity_mat)
    lambdas, _       = eigDecompose(laplacian, is_cuda)
    lambdas          = np.sort(lambdas)
    lambda_gap_list  = getLambdaGapList(lambdas)
    num_of_spk       = np.argmax(
        lambda_gap_list[: min(max_num_speaker, len(lambda_gap_list))]
    ) + 1
    return num_of_spk, lambdas, lambda_gap_list

# ============================================================================
# Enhanced speaker count (anchor embeddings)
# ============================================================================

def addAnchorEmb(emb, anchor_sample_n, anchor_spk_n, sigma):
    emb_dim   = emb.shape[1]
    std_org   = np.std(emb, axis=0)
    new_embs  = []
    for _ in range(anchor_spk_n):
        emb_m     = np.tile(np.random.randn(1, emb_dim), (anchor_sample_n, 1))
        emb_noise = np.random.randn(anchor_sample_n, emb_dim).T
        emb_noise = np.dot(np.diag(std_org),
                           emb_noise / np.max(np.abs(emb_noise))).T
        new_embs.append(emb_m + sigma * emb_noise)
    new_embs.append(emb)
    return np.vstack(new_embs)


def getEnhancedSpeakerCount(emb, cuda,
                             random_test_count=5,
                             anchor_spk_n=3,
                             anchor_sample_n=10,
                             sigma=50):
    est_list = []
    for seed in range(random_test_count):
        np.random.seed(seed)
        emb_aug = addAnchorEmb(emb, anchor_sample_n, anchor_spk_n, sigma)
        mat     = getCosAffinityMatrix(emb_aug)
        nmesc   = NMESC(mat,
                        max_num_speaker=emb.shape[0],
                        max_rp_threshold=0.25,
                        sparse_search=True,
                        sparse_search_volume=30,
                        fixed_thres=None,
                        NME_mat_size=300,
                        cuda=cuda)
        est_num, _ = nmesc.NMEanalysis()
        est_list.append(est_num)
    ctt = Counter(est_list)
    return max(ctt.most_common(1)[0][0] - anchor_spk_n, 1)

# ============================================================================
# NMESC class
# ============================================================================

class NMESC:
    """
    Normalized Maximum Eigengap based Spectral Clustering (NME-SC).

    Reference: https://arxiv.org/abs/2003.02405
    """

    def __init__(self,
                 mat,
                 max_num_speaker=10,
                 max_rp_threshold=0.25,
                 sparse_search=True,
                 sparse_search_volume=30,
                 use_subsampling_for_NME=False,
                 fixed_thres=None,
                 cuda=False,
                 NME_mat_size=512):
        self.mat                    = mat
        self.max_num_speaker        = max_num_speaker
        self.max_rp_threshold       = max_rp_threshold
        self.sparse_search          = sparse_search
        self.sparse_search_volume   = sparse_search_volume
        self.use_subsampling_for_NME = use_subsampling_for_NME
        self.fixed_thres            = fixed_thres
        self.cuda                   = cuda
        self.NME_mat_size           = NME_mat_size
        self.eps                    = 1e-10
        self.max_N                  = None
        self.p_value_list           = []

    def NMEanalysis(self):
        if self.use_subsampling_for_NME:
            subsample_ratio = self.subsampleAffinityMat(self.NME_mat_size)
        else:
            subsample_ratio = 1

        eig_ratio_list  = []
        est_spk_n_dict  = {}
        self.p_value_list = self.getPvalueList()

        print(f"Scanning eig_ratio of length [{len(self.p_value_list)}] "
              f"mat size [{self.mat.shape[0]}] ...")

        for p_value in self.p_value_list:
            est_num_of_spk, g_p = self.getEigRatio(p_value)
            est_spk_n_dict[p_value] = est_num_of_spk
            eig_ratio_list.append(g_p)

        index_nn    = np.argmin(eig_ratio_list)
        rp_p_value  = self.p_value_list[index_nn]
        affinity_mat = getAffinityGraphMat(self.mat, rp_p_value)

        if not isGraphFullyConnected(affinity_mat):
            affinity_mat, rp_p_value = getMinimumConnection(
                self.mat, self.max_N, self.p_value_list)

        p_hat_value    = int(subsample_ratio * rp_p_value)
        est_num_of_spk = est_spk_n_dict[rp_p_value]
        return est_num_of_spk, p_hat_value

    def subsampleAffinityMat(self, NME_mat_size):
        subsample_ratio = int(max(1, self.mat.shape[0] / NME_mat_size))
        self.mat = self.mat[::subsample_ratio, ::subsample_ratio]
        return subsample_ratio

    def getEigRatio(self, p_neighbors):
        affinity_mat = getAffinityGraphMat(self.mat, p_neighbors)
        est_num_of_spk, lambdas, lambda_gap_list = estimateNumofSpeakers(
            affinity_mat, self.max_num_speaker, self.cuda)
        arg_sorted_idx = np.argsort(
            lambda_gap_list[: self.max_num_speaker])[::-1]
        max_key     = arg_sorted_idx[0]
        max_eig_gap = lambda_gap_list[max_key] / (max(lambdas) + self.eps)
        g_p         = (p_neighbors / self.mat.shape[0]) / (max_eig_gap + self.eps)
        return est_num_of_spk, g_p

    def getPvalueList(self):
        if self.fixed_thres is not None:
            p_value_list  = [int(self.mat.shape[0] * self.fixed_thres)]
            self.max_N    = p_value_list[0]
        else:
            self.max_N = int(self.mat.shape[0] * self.max_rp_threshold)
            if self.sparse_search:
                N            = min(self.max_N, self.sparse_search_volume)
                p_value_list = list(
                    np.linspace(1, self.max_N, N, endpoint=True).astype(int))
            else:
                p_value_list = list(range(1, self.max_N))
        return p_value_list

# ============================================================================
# Spectral clustering (k-means on spectral embeddings)
# ============================================================================

class SpectralClustering:
    def __init__(self, n_clusters=8, random_state=0, n_init=10,
                 p_value=10, cuda=False):
        self.n_clusters   = n_clusters
        self.random_state = random_state
        self.n_init       = n_init
        self.p_value      = p_value
        self.cuda         = cuda

    def predict(self, X):
        if X.shape[0] != X.shape[1]:
            raise ValueError("Affinity matrix must be square.")
        return self._clusterSpectralEmbeddings(X)

    def _clusterSpectralEmbeddings(self, affinity):
        spectral_emb = self._getSpectralEmbeddings(
            affinity, n_spks=self.n_clusters, cuda=self.cuda)
        _, labels, _ = k_means(spectral_emb, self.n_clusters,
                                random_state=self.random_state,
                                n_init=self.n_init)
        return labels

    def _getSpectralEmbeddings(self, affinity_mat, n_spks=8,
                                drop_first=True, cuda=False):
        if not isGraphFullyConnected(affinity_mat):
            print("  [WARNING] Graph is not fully connected; "
                  "clustering result might be inaccurate.")
        laplacian        = getLaplacian(affinity_mat)
        lambdas_, diff_  = eigDecompose(laplacian, cuda)
        diffusion_map    = diff_[:, :n_spks]
        embedding        = diffusion_map.T[n_spks::-1]
        return embedding[:n_spks].T

# ============================================================================
# Per-recording clustering entry point
# ============================================================================

def cluster_recording(file_id, emb, metadata,
                      max_speaker=8,
                      spt_est_thres='NMESC',
                      threshold=None,
                      oracle_num_spk=None,
                      min_samples_for_NMESC=6,
                      enhanced_count_thres=80,
                      max_rp_threshold=0.25,
                      sparse_search_volume=30,
                      cuda=False,
                      idx=1):
    """
    Run NME-SC on a single recording.

    Parameters
    ----------
    file_id         : str   – recording identifier
    emb             : np.ndarray (N, D) – raw (non-normalised) embeddings
    metadata        : list[dict] – segment metadata (segment_id, start, end …)
    max_speaker     : int   – upper bound on speaker count
    spt_est_thres   : str   – 'NMESC' | 'None' | path to per-utt thres file
    threshold       : float | None – fixed cosine-similarity pruning threshold
    oracle_num_spk  : int | None   – oracle speaker count if known
    cuda            : bool
    idx             : int   – 1-based index for console output

    Returns
    -------
    labels : np.ndarray (N,) of int speaker IDs
    """
    N = emb.shape[0]

    # ── build cosine affinity matrix ────────────────────────────────────────
    mat = getCosAffinityMatrix(emb)

    # ── resolve fixed_thres ─────────────────────────────────────────────────
    fixed_thres = None
    if spt_est_thres not in ('NMESC', 'None', None) and \
            os.path.isfile(spt_est_thres):
        # per-utterance threshold file: "<utt_id> <thres>" per line
        with open(spt_est_thres) as f:
            thres_map = {l.split()[0]: float(l.split()[1])
                         for l in f if l.strip()}
        fixed_thres = thres_map.get(file_id, None)
    elif threshold is not None:
        fixed_thres = threshold

    # ── enhanced speaker count for short recordings ─────────────────────────
    est_num_of_spk_enhanced = None
    if oracle_num_spk is None and N <= max(enhanced_count_thres,
                                           min_samples_for_NMESC):
        est_num_of_spk_enhanced = getEnhancedSpeakerCount(emb, cuda)

    if oracle_num_spk:
        max_speaker = oracle_num_spk

    # ── NME-SC ──────────────────────────────────────────────────────────────
    nmesc = NMESC(mat,
                  max_num_speaker=max_speaker,
                  max_rp_threshold=max_rp_threshold,
                  sparse_search=True,
                  sparse_search_volume=sparse_search_volume,
                  fixed_thres=fixed_thres,
                  NME_mat_size=300,
                  cuda=cuda)

    if N > min_samples_for_NMESC:
        est_num_of_spk, p_hat_value = nmesc.NMEanalysis()
        affinity_mat = getAffinityGraphMat(mat, p_hat_value)
    else:
        affinity_mat    = mat
        est_num_of_spk  = oracle_num_spk or 1
        p_hat_value     = 1

    # ── resolve final speaker count ─────────────────────────────────────────
    if oracle_num_spk:
        final_num_spk = oracle_num_spk
        prune_label   = f"Rank based pruning - RP threshold: {fixed_thres:.4f}" \
                        if fixed_thres is not None else "Rank based pruning"
        print(f"{idx:2d}  score_metric: cos  {prune_label}"
              f"  key: {file_id}"
              f"  Given Number of Speakers (reco2num_spk): {final_num_spk}"
              f"  MAT size :  {mat.shape}")
    elif est_num_of_spk_enhanced is not None:
        final_num_spk = est_num_of_spk_enhanced
        thresh_label  = (f"threshold: {fixed_thres:.3f}"
                         if fixed_thres is not None
                         else f"threshold: {max_rp_threshold:.3f}")
        print(f"{idx:2d}  score_metric: cos"
              f"  affinity matrix pruning - {thresh_label}"
              f"  key: {file_id}"
              f"  Est # spk: {final_num_spk}"
              f"  Max # spk: {max_speaker}"
              f"  MAT size :  {mat.shape}")
    else:
        final_num_spk = est_num_of_spk
        thresh_label  = (f"threshold: {fixed_thres:.3f}"
                         if fixed_thres is not None
                         else f"threshold: {max_rp_threshold:.3f}")
        print(f"{idx:2d}  score_metric: cos"
              f"  affinity matrix pruning - {thresh_label}"
              f"  key: {file_id}"
              f"  Est # spk: {final_num_spk}"
              f"  Max # spk: {max_speaker}"
              f"  MAT size :  {mat.shape}")

    # ── spectral clustering ─────────────────────────────────────────────────
    spectral_model = SpectralClustering(n_clusters=final_num_spk, cuda=cuda)
    labels         = spectral_model.predict(affinity_mat)
    return labels

# ============================================================================
# I/O helpers
# ============================================================================

def load_recording(npz_path, json_path):
    """Load embeddings and metadata for one recording."""
    data = np.load(npz_path)
    # Support both plain array saves and named-key saves
    if isinstance(data, np.ndarray):
        emb = data
    else:
        keys = list(data.keys())
        emb  = data[keys[0]]          # take the first (and usually only) array

    with open(json_path) as f:
        metadata = json.load(f)

    return emb, metadata


def write_labels(output_dir, file_id, metadata, labels):
    """
    Write per-segment speaker labels to a text file.

    Format (Kaldi-style):
        <segment_id>  <speaker_label>
    """
    out_path = Path(output_dir) / f"{file_id}_labels.txt"
    with open(out_path, "w") as f:
        for seg, label in zip(metadata, labels):
            f.write(f"{seg['segment_id']}  spk_{label:02d}\n")
    return out_path


def write_rttm(output_dir, file_id, metadata, labels):
    """
    Write RTTM file for the recording, applying the midpoint overlap resolution
    from Auto-Tuning-Spectral-Clustering/sc_utils/make_rttm.py.

    RTTM format:
        SPEAKER <file_id> 1 <start> <dur> <NA> <NA> <spk_label> <NA> <NA>
    """
    out_path = Path(output_dir) / f"{file_id}.rttm"
    if not metadata:
        with open(out_path, "w") as f:
            pass
        return out_path

    # Construct initial segments list
    segs = []
    for seg, label in zip(metadata, labels):
        segs.append([float(seg["start"]), float(seg["end"]), f"spk_{label:02d}"])

    # Cut up overlapping segments so they are contiguous at the midpoint
    new_segs = []
    if len(segs) > 0:
        for i in range(len(segs) - 1):
            start, end, label = segs[i]
            next_start, next_end, next_label = segs[i+1]
            if end > next_start:
                avg = (next_start + end) / 2.0
                segs[i+1][0] = avg
                new_segs.append([start, avg, label])
            else:
                new_segs.append([start, end, label])
        new_segs.append(segs[-1])

    # Merge contiguous segments of the same label
    merged_segs = []
    if len(new_segs) > 0:
        current_seg = new_segs[0]
        for i in range(1, len(new_segs)):
            start, end, label = current_seg
            next_start, next_end, next_label = new_segs[i]
            if end == next_start and label == next_label:
                current_seg = [start, next_end, label]
            else:
                merged_segs.append(current_seg)
                current_seg = new_segs[i]
        merged_segs.append(current_seg)

    with open(out_path, "w") as f:
        for start, end, label in merged_segs:
            dur = end - start
            if dur > 0:
                f.write(f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} "
                        f"<NA> <NA> {label} <NA> <NA>\n")
    return out_path

# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="NME-SC speaker diarization clustering "
                    "(tango4j/Auto-Tuning-Spectral-Clustering)")

    p.add_argument("--embeddings_dir", default="Give the embeddings directory path",
                   help="Directory containing *_embeddings.npz and "
                        "*_metadata.json files.")
    p.add_argument("--output_dir", default="Give the output directory path",
                   help="Directory to write label/RTTM files.")
    p.add_argument("--max_speaker", type=int, default=25,
                   help="Maximum number of speakers (default: 8).")
    p.add_argument("--spt_est_thres", default= 'NMESC',
                   help="Speaker estimation threshold mode: "
                        "'NMESC' (auto-tune), 'None' (use --threshold), "
                        "or path to per-utterance threshold file.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Fixed cosine-similarity pruning threshold. "
                        "Used when --spt_est_thres is 'None'.")
    p.add_argument("--reco2num_spk", default=None,
                   help="Path to oracle speaker count file "
                        "(<file_id> <num_spk> per line), or 'None'.")
    p.add_argument("--max_rp_threshold", type=float, default=0.25,
                   help="Max ratio for p-value search range (default: 0.25).")
    p.add_argument("--sparse_search_volume", type=int, default=20,
                   help="Number of p-values to try during NME scan (default: 30).")
    p.add_argument("--enhanced_count_thres", type=int, default=80,
                   help="Segment count below which enhanced speaker count "
                        "is used (default: 80).")
    p.add_argument("--cuda", action="store_true",
                   help="Use CUDA for eigen decomposition if available.")
    p.add_argument("--file_id", default=None,
                   help="Process a single file_id only (optional).")

    return p.parse_args()


def main():
    args = parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load oracle speaker counts ──────────────────────────────────────────
    reco2num_spk = {}
    if args.reco2num_spk and args.reco2num_spk.lower() != "none" \
            and os.path.isfile(args.reco2num_spk):
        print(f"Loading reco2num_spk file:  {args.reco2num_spk}")
        with open(args.reco2num_spk) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    reco2num_spk[parts[0]] = int(parts[1])

    # ── discover recordings ─────────────────────────────────────────────────
    npz_files = sorted(embeddings_dir.glob("*_embeddings.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No *_embeddings.npz files found in {embeddings_dir}")

    if args.file_id:
        npz_files = [f for f in npz_files
                     if f.stem.replace("_embeddings", "") == args.file_id]

    # ── resolve fixed threshold ─────────────────────────────────────────────
    threshold = None
    if args.spt_est_thres.lower() == "none" and args.threshold is not None:
        threshold = args.threshold

    # ── process each recording ──────────────────────────────────────────────
    all_results = {}
    for idx, npz_path in enumerate(npz_files, start=1):
        file_id   = npz_path.stem.replace("_embeddings", "")
        json_path = npz_path.parent / f"{file_id}_metadata.json"

        if not json_path.exists():
            print(f"[SKIP] {file_id}: metadata JSON not found.")
            continue

        emb, metadata = load_recording(str(npz_path), str(json_path))

        if emb.shape[0] != len(metadata):
            print(f"[WARN] {file_id}: embedding rows ({emb.shape[0]}) "
                  f"!= metadata segments ({len(metadata)}). Skipping.")
            continue

        oracle_num_spk = reco2num_spk.get(file_id, None)

        labels = cluster_recording(
            file_id          = file_id,
            emb              = emb,
            metadata         = metadata,
            max_speaker      = args.max_speaker,
            spt_est_thres    = args.spt_est_thres,
            threshold        = threshold,
            oracle_num_spk   = oracle_num_spk,
            enhanced_count_thres = args.enhanced_count_thres,
            max_rp_threshold = args.max_rp_threshold,
            sparse_search_volume = args.sparse_search_volume,
            cuda             = args.cuda,
            idx              = idx,
        )

        label_path = write_labels(str(output_dir), file_id, metadata, labels)
        rttm_path  = write_rttm(str(output_dir), file_id, metadata, labels)
        all_results[file_id] = labels

    print("\nMethod: Spectral Clustering has been finished")
    print(f"Labels  written to: {output_dir}/*_labels.txt")
    print(f"RTTM    written to: {output_dir}/*.rttm")
    return all_results


if __name__ == "__main__":
    main()