#!/usr/bin/env python3
"""
run_pipeline.py — Run the speechbrain ECAPA-TDNN + NME-SC diarization pipeline
over one or more test datasets and collect a per-dataset DER summary CSV.

Pipeline per dataset (calls the existing scripts in this directory,
unchanged, as subprocesses):
    1. ecapa_inf_extractor.py  audio_dir + rttm_dir  -> oracle-VAD segment
                              embeddings (npz/json), one embedding per
                              sliding window over the reference speech regions,
                              using a speechbrain ECAPA-TDNN encoder
    2. cos+sc.py               embeddings            -> hypothesis RTTMs
                              (cosine-affinity NME spectral clustering)
    3. der.py                  hypothesis vs. rttm_dir (ground truth) -> DER
                              (overlap excluded, 0.25 s collar by default)

Output layout:
    <output_root>/
        der_summary.csv              <- one row per dataset
        <dataset_name>/
            embeddings/               *_embeddings.npz, *_metadata.json,
                                      *_affinity.npy
            hyp_rttm/                 *_labels.txt, *.rttm   (system output)
            der_report.txt            full pyannote.metrics report
            speaker_count.csv         true (reference) vs. predicted number
                                      of speakers per recording
            logs/extract.log
            logs/cluster.log
            logs/der.log

Usage:
    # run every dataset in DATASETS below
    python run_pipeline.py --output_root ./runs/v1_cos_sc

    # run only a subset
    python run_pipeline.py --output_root ./runs/v1_cos_sc \\
        --datasets ami_test voxconverse_test

    # re-run everything even if outputs already exist
    python run_pipeline.py --output_root ./runs/v1_cos_sc --force

Resumability:
    A dataset counts as done only once DER has been computed successfully;
    that is recorded in <dataset>/.pipeline_done. On the next run:
      - dataset has .pipeline_done            -> skipped, cached row reused
      - dataset dir exists but no marker       -> treated as a partial/
        interrupted run, the whole <dataset> dir is deleted and it is
        re-run from scratch (extraction + clustering + DER)
      - dataset dir doesn't exist yet          -> run from scratch
    --force ignores .pipeline_done too and always redoes everything.
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from pyannote.database.util import load_rttm

V1_DIR = Path(__file__).resolve().parent
EXTRACTOR = V1_DIR / "ecapa_inf_extractor.py"
CLUSTERER = V1_DIR / "cos+sc.py"
DER_SCRIPT = V1_DIR / "der.py"
DEFAULT_OUTPUT_ROOT = V1_DIR / "v1_output"

# ============================================================================
# CONFIG —
# Each entry needs a flat audio_dir (*.wav/*.flac) and a flat rttm_dir
# (*.rttm) where matching files share the same stem (file_id).
# ============================================================================

# ── datasets to evaluate ────────────────────────────────────────────────────
DATASET_REGISTRY = {
    # "voxconverse_test": {
    #     "audio_dir": "./diarisation_dataset/voxconverse/test/wav",
    #     "rttm_dir": "./diarisation_dataset/voxconverse/test/rttm",
    # },
    # "dihard_second": {
    #     "audio_dir": "./dihard-datasets/second_dihard_challenge_eval-eleven_src/data/single_channel/flac",
    #     "rttm_dir": "./dihard-datasets/second_dihard_challenge_eval-eleven_src/data/single_channel/rttm",
    # },
    # "dihard_third": {
    #     "audio_dir": "./dihard-datasets/third_dihard_challenge_eval/data/flac",
    #     "rttm_dir": "./dihard-datasets/third_dihard_challenge_eval/data/rttm",
    # },
    # "alimeeting_far": {
    #     "audio_dir": "./diarisation_dataset/alimeeting_first_channel_far/wav/Test_Ali_far",
    #     "rttm_dir": "./diarisation_dataset/alimeeting_first_channel_far/rttm/Test_Ali_far",
    # },
    # "alimeeting_near": {
    #     "audio_dir": "./diarisation_dataset/alimeeting_near/wav/Eval_Test_near/Test_Ali_near",
    #     "rttm_dir": "./diarisation_dataset/alimeeting_near/rttm/Eval_Test_near/Test_Ali_near",
    # },
    # "ami_sdm1": {
    #     "audio_dir": "./diarisation_dataset/ami_sdm1_split/test/audio",
    #     "rttm_dir": "./diarisation_dataset/ami_sdm1_split/test/rttm",
    # },
    # "AISHELL-4": {
    #     "audio_dir": "./diarisation_dataset/AISHELL-4/test/wav",
    #     "rttm_dir": "./diarisation_dataset/AISHELL-4/test/TextGrid",
    # },
    "AMI_mixheadset": {
        "audio_dir": "./diarisation_dataset/ami_mixed/audio_split/test",
        "rttm_dir": "./diarisation_dataset/ami_mixed/BUT_rttm/test",
    }

}

# ── ECAPA model + embedding extraction ──────────────────────────────────────
MODEL = "speechbrain/spkrec-ecapa-voxceleb"  # speechbrain ECAPA-TDNN
WINDOW = 1.5                                    # sliding-window length (s)
SHIFT = 0.75                                    # sliding-window hop (s)
SAMPLE_RATE = 16000
AUDIO_EXT = ".wav"                              # extractor also tries .flac/.wav

# ── clustering (cos+sc.py) ──────────────────────────────────────────────────
MAX_SPEAKER = 25

# ── DER scoring (der.py) ────────────────────────────────────────────────────
COLLAR = 0.25          # forgiveness collar in seconds
SKIP_OVERLAP = True    # exclude overlapping speech regions from scoring

# ============================================================================

OVERALL_DER_RE = re.compile(r"Overall DER\s*:\s*([\d.]+)%")
FILES_EVAL_RE = re.compile(r"Files evaluated:\s*(\d+)\s*/\s*(\d+)")

SPEAKER_COUNT_FIELDS = ["file_id", "true_num_speakers",
                        "predicted_num_speakers", "difference"]


def write_speaker_count_csv(ref_dir, hyp_dir, csv_path):
    """One row per recording: reference speaker count vs. predicted
    speaker-cluster count (e.g. true=4, predicted=6, difference=+2)."""
    rows = []
    for ref_file in sorted(Path(ref_dir).glob("*.rttm")):
        stem = ref_file.stem
        hyp_file = Path(hyp_dir) / f"{stem}.rttm"
        if not hyp_file.exists():
            continue

        ref_dict = load_rttm(str(ref_file))
        hyp_dict = load_rttm(str(hyp_file))
        if not ref_dict or not hyp_dict:
            continue

        reference = ref_dict[list(ref_dict.keys())[0]]
        hypothesis = hyp_dict[list(hyp_dict.keys())[0]]

        n_true = len(reference.labels())
        n_pred = len(hypothesis.labels())
        rows.append({
            "file_id": stem,
            "true_num_speakers": n_true,
            "predicted_num_speakers": n_pred,
            "difference": n_pred - n_true,
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SPEAKER_COUNT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def run_step(cmd, log_path):
    """Run a subprocess, tee output to a log file, return (returncode, stdout_text)."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path.write_text(proc.stdout)
    if proc.returncode != 0:
        print(f"    [FAILED] see {log_path}")
    return proc.returncode, proc.stdout


def run_dataset(name, audio_dir, rttm_dir, out_root, args):
    ds_dir = out_root / name
    marker_path = ds_dir / ".pipeline_done"
    embeddings_dir = ds_dir / "embeddings"
    hyp_dir = ds_dir / "hyp_rttm"
    logs_dir = ds_dir / "logs"
    der_report = ds_dir / "der_report.txt"

    # ── already fully evaluated in a previous run? reuse it as-is ──────────
    if not args.force and marker_path.exists():
        cached = json.loads(marker_path.read_text())
        print(f"\n=== Dataset: {name} === "
              f"[SKIP] already fully evaluated (DER={cached.get('overall_der_percent')}%); "
              f"use --force to redo")
        return cached

    print(f"\n=== Dataset: {name} ===")

    # ── leftover dir with no completion marker == a partial/interrupted
    #    previous run (or --force). Wipe it and start that dataset clean. ──
    if ds_dir.exists():
        reason = "--force" if args.force else "partial/incomplete output found"
        print(f"  [CLEAN] {reason}; removing {ds_dir} and re-running from scratch")
        shutil.rmtree(ds_dir)

    row = {
        "dataset": name,
        "audio_dir": audio_dir,
        "rttm_dir": rttm_dir,
        "overall_der_percent": "",
        "files_evaluated": "",
        "total_ref_files": "",
        "status": "OK",
    }

    if not Path(audio_dir).is_dir():
        print(f"  [SKIP] audio_dir not found: {audio_dir}")
        row["status"] = "MISSING_AUDIO_DIR"
        return row
    if not Path(rttm_dir).is_dir():
        print(f"  [SKIP] rttm_dir not found: {rttm_dir}")
        row["status"] = "MISSING_RTTM_DIR"
        return row

    ds_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. embedding extraction (oracle VAD + sliding-window ECAPA) ────────
    cmd = [
        sys.executable, str(EXTRACTOR),
        "--model", args.model,
        "--audio_dir", str(audio_dir),
        "--rttm_dir", str(rttm_dir),
        "--out_dir", str(embeddings_dir),
        "--window", str(args.window),
        "--shift", str(args.shift),
        "--sr", str(args.sr),
        "--audio_ext", args.audio_ext,
    ]
    rc, _ = run_step(cmd, logs_dir / "extract.log")
    if rc != 0:
        row["status"] = "EXTRACTION_FAILED"
        return row

    # ── 2. clustering ────────────────────────────────────────────────────
    cmd = [
        sys.executable, str(CLUSTERER),
        "--embeddings_dir", str(embeddings_dir),
        "--output_dir", str(hyp_dir),
        "--max_speaker", str(args.max_speaker),
    ]
    rc, _ = run_step(cmd, logs_dir / "cluster.log")
    if rc != 0:
        row["status"] = "CLUSTERING_FAILED"
        return row

    # ── 3. DER (overlap excluded, collar applied) ───────────────────────────
    cmd = [
        sys.executable, str(DER_SCRIPT),
        "--ref_dir", str(rttm_dir),
        "--hyp_dir", str(hyp_dir),
        "--out", str(der_report),
        "--collar", str(args.collar),
    ]
    if not args.skip_overlap:
        cmd.append("--no_skip_overlap")
    rc, stdout = run_step(cmd, logs_dir / "der.log")
    if rc != 0:
        row["status"] = "DER_FAILED"
        return row

    m_der = OVERALL_DER_RE.search(stdout)
    m_files = FILES_EVAL_RE.search(stdout)
    if not m_der:
        row["status"] = "DER_NOT_FOUND"
        return row

    row["overall_der_percent"] = m_der.group(1)
    if m_files:
        row["files_evaluated"] = m_files.group(1)
        row["total_ref_files"] = m_files.group(2)
    print(f"  DER = {row['overall_der_percent']}%  "
          f"({row['files_evaluated']}/{row['total_ref_files']} files)")

    # ── 4. per-recording true-vs-predicted speaker count CSV ───────────────
    count_csv = ds_dir / "speaker_count.csv"
    try:
        n_rows = write_speaker_count_csv(rttm_dir, hyp_dir, count_csv)
        print(f"  speaker count: {n_rows} recordings -> {count_csv}")
    except Exception as e:
        print(f"  [WARN] failed to write speaker count CSV: {e}")

    # only reached on full success -> safe to mark this dataset as done
    marker_path.write_text(json.dumps(row, indent=2))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT),
                   help="Root output directory (created if missing). "
                        f"Default: {DEFAULT_OUTPUT_ROOT}")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Subset of DATASETS/JSON dataset names to run "
                        "(default: run all).")
    p.add_argument("--datasets_json", default=None,
                   help="JSON file of {name: {audio_dir, rttm_dir}} entries "
                        "merged on top of the built-in DATASETS config.")
    p.add_argument("--model", default=MODEL,
                   help="ECAPA source passed to ecapa_inf_extractor.py's --model "
                        "(a speechbrain ECAPA-TDNN model dir/hub id).")
    p.add_argument("--window", type=float, default=WINDOW,
                   help="Sliding-window length (s) passed to the extractor.")
    p.add_argument("--shift", type=float, default=SHIFT,
                   help="Sliding-window hop (s) passed to the extractor.")
    p.add_argument("--sr", type=int, default=SAMPLE_RATE,
                   help="Sample rate passed to the extractor.")
    p.add_argument("--audio_ext", default=AUDIO_EXT,
                   help="Audio extension passed to the extractor's --audio_ext "
                        "(it also auto-tries .flac/.wav regardless).")
    p.add_argument("--max_speaker", type=int, default=MAX_SPEAKER,
                   help="Max speakers passed to cos+sc.py.")
    p.add_argument("--collar", type=float, default=COLLAR,
                   help="DER collar in seconds.")
    p.add_argument("--skip_overlap", dest="skip_overlap", action="store_true",
                   default=SKIP_OVERLAP,
                   help="Exclude overlapping speech from DER scoring (default).")
    p.add_argument("--no_skip_overlap", dest="skip_overlap", action="store_false",
                   help="Score overlapping speech too.")
    p.add_argument("--force", action="store_true",
                   help="Wipe and re-run every dataset from scratch, even "
                        "ones already marked fully evaluated.")
    args = p.parse_args()

    datasets = dict(DATASET_REGISTRY)
    if args.datasets_json:
        with open(args.datasets_json) as f:
            datasets.update(json.load(f))

    if args.datasets:
        missing = [d for d in args.datasets if d not in datasets]
        if missing:
            p.error(f"Unknown dataset name(s): {missing}. "
                    f"Available: {sorted(datasets)}")
        datasets = {k: datasets[k] for k in args.datasets}

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, ds in datasets.items():
        row = run_dataset(name, ds["audio_dir"], ds["rttm_dir"], out_root, args)
        rows.append(row)

    csv_path = out_root / "der_summary.csv"
    fieldnames = ["dataset", "overall_der_percent", "files_evaluated",
                  "total_ref_files", "status", "audio_dir", "rttm_dir"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== Summary written to {csv_path} ===")
    for row in rows:
        print(f"  {row['dataset']:<20} DER={row['overall_der_percent'] or 'N/A':>6}%  "
              f"status={row['status']}")


if __name__ == "__main__":
    main()