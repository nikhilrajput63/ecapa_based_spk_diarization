#!/usr/bin/env python3
"""
DER Evaluation Script - Baseline AHC
=====================================
Computes Diarization Error Rate (DER) by comparing predicted RTTM files
against ground-truth reference RTTMs using pyannote.metrics.

Usage:
    python der.py                         # use default paths (Config below)
    python der.py --hyp_dir ./output_ami/ecapa/ahc --out ./der_report.txt
"""

import os
import glob
import argparse
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

# ==============================================================================
# CONFIGURATION PARAMETERS AND PATHS
# ==============================================================================
class Config:
    # Ground Truth (Reference) RTTM directory
    REF_RTTM_DIR = "/DATA/nikhil-data/diarisation_dataset/ami_mixed/BUT_rttms/test"

    # Predicted (Hypothesis) RTTM directory  ← change this as needed
    HYP_RTTM_DIR = "./ecapa/output_ami/cos+sc"

    # Optional: path to save the detailed report (set to None to skip saving)
    OUTPUT_REPORT = "./ecapa/output_ami/der_report_ecapa_cos+sc.txt"

    # DER settings
    SKIP_OVERLAP = True   # ignore overlapping speech regions (standard for AMI)
    COLLAR       = 0.25   # forgiveness collar in seconds (standard: 0.25 s)


# ==============================================================================
# EVALUATION LOGIC
# ==============================================================================

def evaluate_rttms(ref_dir: str, hyp_dir: str, output_report: str,
                   skip_overlap: bool, collar: float):
    print(f"Reference RTTMs : {ref_dir}")
    print(f"Hypothesis RTTMs: {hyp_dir}\n") 

    ref_files = sorted(glob.glob(os.path.join(ref_dir, "*.rttm")))
    if not ref_files:
        print("[ERROR] No reference RTTM files found!")
        return

    metric = DiarizationErrorRate(skip_overlap=skip_overlap, collar=collar)
    evaluated_files = 0

    for ref_file in ref_files:
        basename = os.path.basename(ref_file)
        stem     = os.path.splitext(basename)[0]
        hyp_file = os.path.join(hyp_dir, basename)

        if not os.path.exists(hyp_file):
            print(f"[WARNING] No hypothesis RTTM found for {stem}. Skipping.")
            continue

        try:
            ref_dict = load_rttm(ref_file)
            hyp_dict = load_rttm(hyp_file)

            if not ref_dict or not hyp_dict:
                print(f"[WARNING] Empty RTTM for {stem}. Skipping.")
                continue

            reference  = ref_dict[list(ref_dict.keys())[0]]
            hypothesis = hyp_dict[list(hyp_dict.keys())[0]]

            file_der = abs(metric(reference, hypothesis))
            print(f"  [{stem}]  DER = {file_der:.2%}")
            evaluated_files += 1

        except Exception as e:
            print(f"[ERROR] Failed to process {stem}: {e}")

    if evaluated_files == 0:
        print("[ERROR] No valid RTTM pairs were evaluated.")
        return

    print("\n" + "=" * 60)
    print("FINAL EVALUATION REPORT")
    print("=" * 60)
    overall_der = abs(metric)
    print(f"Overall DER    : {overall_der:.2%}")
    print(f"Files evaluated: {evaluated_files} / {len(ref_files)}")
    print(f"Settings       : collar={collar}s, skip_overlap={skip_overlap}\n")

    report_df = metric.report(display=True)

    if output_report:
        os.makedirs(os.path.dirname(output_report), exist_ok=True)
        with open(output_report, "w") as f:
            f.write("FINAL EVALUATION REPORT\n")
            f.write(f"Overall DER: {overall_der:.2%}\n")
            f.write(f"Files evaluated: {evaluated_files} / {len(ref_files)}\n")
            f.write(f"Settings: collar={collar}s, skip_overlap={skip_overlap}\n\n")
            f.write(report_df.to_string())
        print(f"\nReport saved to: {output_report}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute DER for speaker diarization RTTM outputs.")
    parser.add_argument("--ref_dir", type=str, default=Config.REF_RTTM_DIR,
        help="Directory containing ground-truth reference RTTM files.")
    parser.add_argument("--hyp_dir", type=str, default=Config.HYP_RTTM_DIR,
        help="Directory containing predicted hypothesis RTTM files.")
    parser.add_argument("--out", type=str, default=Config.OUTPUT_REPORT,
        help="Path to save the detailed DER report (or empty to skip saving).")
    parser.add_argument("--collar", type=float, default=Config.COLLAR,
        help="Forgiveness collar in seconds (default: 0.25).")
    parser.add_argument("--no_skip_overlap", action="store_true",
        help="If set, do NOT skip overlapping speech regions (default: skip).")

    args = parser.parse_args()
    evaluate_rttms(
        ref_dir      = args.ref_dir,
        hyp_dir      = args.hyp_dir,
        output_report= args.out,
        skip_overlap = not args.no_skip_overlap,
        collar       = args.collar,
    )
