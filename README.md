# Baseline: ECAPA-TDNN + NME-SC Speaker Diarization
```text
audio + reference RTTM (oracle VAD)
  → ECAPA embeddings   1.5 s window / 0.75 s hop, 16 kHz
  → cosine affinity → NME-SC (eigengap picks k) → k-means
  → hypothesis RTTM → DER (collar 0.25 s, overlap skipped)
```
Clustering follows Park et al., *Auto-Tuning Spectral Clustering for Speaker
Diarization Using Normalized Maximum Eigengap* ([arXiv:2003.02405](https://arxiv.org/abs/2003.02405)),
adapted from `tango4j/Auto-Tuning-Spectral-Clustering` and NVIDIA NeMo.

## Install

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
## Usage

Edit `DATASET_REGISTRY` in `calculate_various_der.py` (entries need a flat
`audio_dir` and `rttm_dir` sharing file stems), then:

```bash
python calculate_various_der.py --output_root ./runs/baseline
```
NME-SC diagnostic plots (separate from the driver):

```bash
python vis_num_spk.py --embeddings_dir ./runs/baseline/<dataset>/embeddings \
                      --output_dir ./nmesc_plots
```


## Files

| File | Role |
| :--- | :--- |
| **`calculate_various_der.py`** | **Driver** — runs stages 1→3 per dataset, writes `der_summary.csv` |
| `ecapa_inf_extractor.py` | Stage&nbsp;1 — oracle-VAD sliding-window ECAPA embeddings |
| `cos+sc.py` | Stage&nbsp;2 — NME-SC clustering → hypothesis RTTMs |
| `der.py` | Stage&nbsp;3 — DER via `pyannote.metrics` |
| `vis_num_spk.py` | Diagnostic — eigengap plots |


## Output

```text
<output_root>/
├── der_summary.csv
└── <dataset>/
    ├── embeddings/   hyp_rttm/   logs/
    ├── der_report.txt
    ├── speaker_count.csv     # true vs. predicted speaker count per recording
    └── .pipeline_done
```
## Model

`--model` takes a local directory or a HuggingFace id
(`speechbrain/spkrec-ecapa-voxceleb`, 192-D), downloaded on first use into
`pretrained_models/` next to the extractor. Override with `--savedir` — it must
differ from a hub id, or SpeechBrain reads the half-downloaded folder as a local
model and stops fetching.

## Results

`v1_output/`, oracle VAD, collar 0.25 s, overlap excluded:

| Dataset | DER | Files |
| --- | ---: | ---: |
| alimeeting_near | 0.99 % | 20 |
| alimeeting_far | 1.79 % | 20 |
| AMI_mixheadset | 2.54 % | 16 |
| AISHELL-4 | 4.45 % | 20 |
| ami_sdm1 | 4.95 % | 16 |
| voxconverse_test | 5.22 % | 232 |
| dihard_third | 5.50 % | 259 |
| dihard_second | 7.20 % | 171 |

> **Not comparable to published DERs.** The reference RTTM supplies the speech
> regions, so there is no missed-speech or false-alarm error, and overlap is
> excluded. What remains is essentially speaker-confusion error
>  Compare variants under this protocol only.



