"""
extract_embeddings.py   (STAGE 1)

audio + reference RTTM  ->  ECAPA segment embeddings + metadata + cosine affinity.

Per recording <fid>, writes to --out_dir:
  <fid>_embeddings.npz   embedding: (N, D) float32   (NOT L2-normalised)
  <fid>_metadata.json    [{segment_id, file_id, start, end}, ...]
  <fid>_affinity.npy     (N, N) cosine affinity in [0,1]  (consumed by the refiner)

The RTTM is used ONLY for oracle speech activity (where speech is), never for
speaker labels — this is diarisation inference, the labels are for scoring only.
Speech regions are windowed (sliding window) and one ECAPA embedding is taken
per window, matching the segment format your training pipeline expects.

CRITICAL: use the SAME ECAPA model/config that produced your TRAINING embeddings.
The refiner checkpoint expects dim == training dim (512 here); a different model
(e.g. the stock 192-d spkrec-ecapa-voxceleb) will mismatch at refine time. If
your training extractor is custom, swap the `encode_segment` body for it.

Run:
  python extract_embeddings.py \
      --audio_dir /path/icsi/audio --rttm_dir /path/icsi/rttm \
      --out_dir ./icsi_output/embeddings \
      --model speechbrain/spkrec-ecapa-voxceleb \
      --window 1.5 --shift 0.75 --sr 16000 --audio_ext .wav
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ── RTTM -> oracle speech regions (labels ignored) ─────────────────────────────

def parse_rttm_speech(path):
    """Union of all SPEAKER segments -> merged (start, end) speech regions."""
    segs = []
    for line in open(path):
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            s = float(p[3]); d = float(p[4])
            segs.append((s, s + d))
    if not segs:
        return []
    segs.sort()
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def make_windows(regions, win, shift, min_seg=0.5):
    """Sliding windows over speech regions -> list of (start, end)."""
    out = []
    for rs, re in regions:
        if re - rs < min_seg:
            continue
        t = rs
        while t + win <= re + 1e-6:
            out.append((t, t + win))
            t += shift
        # tail window so the end of the region is covered
        if not out or out[-1][1] < re - 1e-6:
            out.append((max(rs, re - win), re))
    return out


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default="Give the audio directory path", help="Directory containing audio files (.wav/.flac) matching the RTTM files.")
    ap.add_argument("--rttm_dir",  default="Give the RTTM directory path", help="Directory containing RTTM files.")
    ap.add_argument("--out_dir",   default="Output directory for embeddings, metadata, and affinity matrices.", help="Directory where the output files will be saved.")
    ap.add_argument("--model",     default="./",
                    help="ECAPA source — MUST match your training extractor (dim 512)")
    ap.add_argument("--savedir",   default=None,
                    help="Where a downloaded model is cached. Default: --model "
                         "itself if it is a local dir, else ./pretrained_models/"
                         "<model> next to this script.")
    ap.add_argument("--window",    type=float, default=1.5)
    ap.add_argument("--shift",     type=float, default=0.75)
    ap.add_argument("--sr",        type=int,   default=16000)
    ap.add_argument("--audio_ext", default=".wav")
    ap.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import torchaudio
    try:
        from speechbrain.inference.speaker import EncoderClassifier   # SB >= 1.0
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier          # older SB

    if args.savedir:
        savedir = args.savedir
    elif Path(args.model).is_dir():
        savedir = args.model
    else:
        savedir = str(Path(__file__).resolve().parent / "pretrained_models"
                      / args.model.replace("/", "_"))

    enc = EncoderClassifier.from_hparams(source=args.model,
                                         savedir=savedir,
                                         run_opts={"device": args.device})

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rttms = sorted(Path(args.rttm_dir).glob("*.rttm"))
    if not rttms:
        raise SystemExit(f"no .rttm files in {args.rttm_dir}")

    min_samp = int(0.3 * args.sr)
    for rt in rttms:
        fid   = rt.stem
        audio = None
        for ext in [args.audio_ext, ".flac", ".wav"]:
            cand = Path(args.audio_dir) / f"{fid}{ext}"
            if cand.exists():
                audio = cand
                break

        if not audio:
            # fall back to a suffixed match, e.g. rttm stem "R8002_M8002" vs.
            # audio file "R8002_M8002_MS802.wav" (extra mic/channel suffix).
            for ext in [args.audio_ext, ".flac", ".wav"]:
                matches = sorted(Path(args.audio_dir).glob(f"{fid}_*{ext}"))
                if matches:
                    if len(matches) > 1:
                        print(f"[warn] {len(matches)} suffixed audio matches for {fid} in "
                              f"{args.audio_dir}, using {matches[0].name}")
                    audio = matches[0]
                    break

        if not audio:
            print(f"[skip] no audio for {fid} in {args.audio_dir} (tried {args.audio_ext}, .flac, .wav, "
                  f"and suffixed variants)")
            continue

        wav, sr = torchaudio.load(str(audio))
        if sr != args.sr:
            wav = torchaudio.functional.resample(wav, sr, args.sr)
        wav = wav.mean(0)                                   # to mono (T,)

        wins = make_windows(parse_rttm_speech(rt), args.window, args.shift)
        embs, meta = [], []
        for i, (s, e) in enumerate(wins):
            seg = wav[int(s * args.sr): int(e * args.sr)]
            if seg.numel() < min_samp:
                continue
            with torch.no_grad():
                ev = enc.encode_batch(seg.unsqueeze(0).to(args.device))  # (1,1,D)
            embs.append(ev.squeeze().detach().cpu().numpy())
            meta.append({"segment_id": f"segment_{i:05d}", "file_id": fid,
                         "start": round(s, 3), "end": round(e, 3)})

        if not embs:
            print(f"[skip] {fid}: no valid segments")
            continue

        E  = np.stack(embs).astype(np.float32)             # (N, D)
        En = F.normalize(torch.from_numpy(E), dim=-1)
        A  = (((En @ En.t()) + 1.0) * 0.5).numpy().astype(np.float32)   # cosine [0,1]

        np.savez(out / f"{fid}_embeddings.npz", embedding=E)
        json.dump(meta, open(out / f"{fid}_metadata.json", "w"))
        np.save(out / f"{fid}_affinity.npy", A)
        print(f"[ok] {fid}: {E.shape[0]} segments | dim {E.shape[1]} -> {out}")

    print(f"\ndone -> {out}\nnext: python refine_affinity.py --ckpt <ckpt> "
          f"--in_dir {out} --out_dir <refined>")


if __name__ == "__main__":
    main()