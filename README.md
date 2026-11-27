# VoiceGuard: AI voice deepfake detector

VoiceGuard tells whether a speech clip is a **real person** or **AI-generated**
(text-to-speech or a voice clone). Voice-clone scams ("Mum, I'm in trouble, send
money") are now cheap to make, and detectors that look perfect on lab benchmarks
often fail on real audio from phones and the internet. This project measures that
gap honestly and works on closing it.

> **Status: work in progress.** Our own trained models (Runs 2 and 3) are still
> being trained and judged on a GPU cluster; the table below is updated as results come in.

## Results so far

Lower is better. **EER** = error rate at the best possible threshold, chosen in
hindsight. **HTER** = the honest number: the threshold is frozen on the
development set *before* seeing the test set, then applied unchanged.

| Model | ASVspoof 2019 eval (lab) EER / HTER | In-the-Wild (real world) EER / HTER |
|---|---|---|
| Baseline: frozen XLS-R + logistic regression | 1.93% / 2.61% | 28.90% / 31.45% |
| Baseline + noise / phone / MP3 augmentation | 1.10% / 1.91% | 26.51% / 32.63% |
| SSL-AASIST, authors' LA model (Tak et al., 2022) | 0.23% / 1.01% | 11.21% / 15.27% |
| SSL-AASIST, authors' DF model | 0.15% / 0.27% | **8.23%** / 25.10% |
| ↳ same, threshold frozen on *messy* dev copies | | 8.23% / **11.71%** |
| **Ours, Run 1**: SSL-AASIST fine-tuned on 2019 + noise/phone/MP3/echo | 1.09% / 2.70% | 22.29% / 23.32% |
| Ours, Run 2: + RawBoost + random 4-s windows | *being judged* | *after selection* |
| Ours, Run 3: + ASVspoof 5 training data (8× more) | *training* | *after selection* |

What the numbers show:

- **Lab scores don't predict real-world scores.** Every model is near-perfect on
  ASVspoof 2019, then 15–55× worse on In-the-Wild. Run 1 reached 0.08% EER on its
  development set and still called a third of real internet speakers fake:
  it learned what 2019 *recordings* sound like, not what *fakes* sound like.
- **A good EER can hide a bad product.** The authors' DF model has 8.23% EER on
  In-the-Wild, but with the threshold chosen on clean lab audio it flags half of
  real speakers as fake (HTER 25.10%). Choosing the threshold on deliberately
  roughed-up dev copies more than halves that (11.71%).
- **Measure carefully.** An early version of our EER code checked only 101
  thresholds; SSL-AASIST's scores sit so close to 0 and 1 that this reported
  24.51% instead of 8.23%. The fix: use every score as a candidate threshold.

## How models are chosen (and why the final score is honest)

Each dataset has exactly one job, and nothing we test on is ever trained on:

1. **Train** (ASVspoof 2019 train; Run 3 adds ASVspoof 5 train): the model learns.
2. **Dev** (ASVspoof 2019 dev): picks the best epoch and freezes the threshold.
3. **Referee** (ASVspoof 2021 DF, a fixed 25,535-clip sample): picks the best *run*.
4. **Final exams**, used once, on the chosen model only:
   In-the-Wild, ML-ITW (14 languages, 7 platforms) and a small set of my own
   recordings with matching TTS and voice-clone fakes.

Using a final exam to choose between models would quietly tune to it, so the
reported score would be better than the truth.

## How it works

- **Front end:** Meta's XLS-R (wav2vec 2.0, 300M parameters, pretrained on 436k
  hours of speech in 128 languages) turns raw 16 kHz audio into features.
- **Back end:** AASIST, a graph-attention network that looks at patterns across
  time and frequency, outputs *real* vs *fake*.
- **Training** (our runs): all 300M XLS-R weights are fine-tuned with a very small
  learning rate (1e-6) so pretraining isn't wrecked; weighted loss for the 9:1
  fake/real imbalance; early stopping on dev EER.
- **Augmentation:** background noise, phone-line band-pass, MP3 compression and
  room echo ([augment.py](src/voiceguard/augment.py)), plus RawBoost
  ([rawboost.py](src/voiceguard/rawboost.py)): random microphone colouring,
  clicks and coloured background noise.

## Repository layout

```
src/voiceguard/   audio loading, augmentation, RawBoost, SSL-AASIST model, dataset readers, metrics
scripts/          feature extraction, training, scoring, evaluation
cluster/          SLURM job and dataset download scripts for the GPU cluster
tests/            unit tests (metrics, dataset readers)
01_explore_audio.ipynb   first look at waveforms and spectrograms
```

## Running it

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run pytest
```

No datasets or model weights are included. Download the datasets yourself (see
below) into `data/`, e.g. `data/asvspoof2019/LA/...` and
`data/in_the_wild/release_in_the_wild/...`. The authors' SSL-AASIST checkpoints
come from [their repository](https://github.com/TakHemlata/SSL_Anti-spoofing)
and go in `models/ssl_aasist/`.

```bash
# Baseline: frozen XLS-R features + logistic regression (runs on a laptop)
uv run python scripts/extract_features.py train      # also dev, eval, in_the_wild
uv run python scripts/train_classifier.py
uv run python scripts/evaluate.py in_the_wild --model logreg

# Score with a pretrained or our own SSL-AASIST, then grade (threshold frozen on dev)
uv run python scripts/score_ssl_aasist.py dev --checkpoint Best_LA_model_for_DF
uv run python scripts/score_ssl_aasist.py in_the_wild --checkpoint Best_LA_model_for_DF
uv run python scripts/evaluate.py in_the_wild --model ssl_aasist_df

# Fine-tune our own SSL-AASIST on one GPU (SLURM)
sbatch cluster/train.sh --out models/run2 --rawboost-prob 0.5 --random-windows
```

## Datasets and credits

| Used for | Dataset | Licence |
|---|---|---|
| Training, dev, lab test | [ASVspoof 2019 LA](https://doi.org/10.7488/ds/2555) | ODC-By |
| Training (Run 3) | [ASVspoof 5](https://zenodo.org/records/14498691) | CC BY 4.0 |
| Referee | [ASVspoof 2021 DF](https://zenodo.org/record/4835108) | see dataset page |
| Final exam | [In-the-Wild](https://deepfake-total.com/in_the_wild) (Müller et al., 2022) | CC BY-NC 4.0 |
| Final exam | ML-ITW (Xue et al., 2026, [arXiv:2603.05852](https://arxiv.org/abs/2603.05852)) | no licence stated; evaluation only |
| Final exam | My own recordings + fakes made with Kokoro-82M (Apache-2.0) and XTTS-v2 (CPML, non-commercial) | not shared |

- Model: SSL-AASIST, Tak et al., "Automatic speaker verification spoofing and
  deepfake detection using wav2vec 2.0 and data augmentation", Odyssey 2022.
  Code ported from [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)
  (MIT) from fairseq to Hugging Face `transformers`.
- RawBoost: Tak et al., ICASSP 2022 (MIT, same repository).
- AASIST: Jung et al., ICASSP 2022. XLS-R: Babu et al., 2021 (Meta AI).

## Limits

- A detector is a risk signal, not proof: it will make mistakes, and new
  voice generators appear faster than test sets.
- Results are on English-heavy benchmarks; ML-ITW is the multilingual check.
- Not new research: this reproduces and extends published work, with an
  emphasis on honest real-world evaluation.

## Licence

Code: MIT, see [LICENSE](LICENSE) (includes the MIT notice for the ported
SSL-AASIST and RawBoost code). Datasets and model weights are not part of this
repository and keep their own licences.
