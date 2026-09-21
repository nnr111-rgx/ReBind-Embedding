# ReBind-Embedding

Official research implementation of **REBIND**, a reconstruction-aware neural embedding framework for sequence edit distance and insertion-deletion-substitution (IDS) channels.

The repository contains two setups:

- **Binary REBIND v1** — reconstruction-aware edit-distance embedding for binary sequences.
- **4-ary setup** — REBIND for 4-ary IDS communication experiments, including whitening, soft-output calibration, and BCJR-guided distillation.

## Repository Structure

```text
ReBind-Embedding/
├── src/rebind/          # Binary REBIND implementation
├── scripts/             # Binary training, evaluation, and plotting utilities
├── tests/               # Binary setup tests
└── 4ary_setup/          # Self-contained 4-ary IDS setup
    ├── src/rebind4ary/
    ├── scripts/
    └── tests/
```

## Binary REBIND

The binary model maps variable-length sequences into a fixed-dimensional embedding while preserving edit-distance structure and supporting sequence reconstruction.

### Inference / Evaluation

```bash
python scripts/eval/03_eval_all.py \
  --h5 <dataset.h5> \
  --split-json <split.json> \
  --checkpoint <checkpoint.pt> \
  --output-dir outputs/binary
```

The evaluation reports edit-distance and reconstruction metrics.

## 4-ary IDS Setup

The `4ary_setup/` folder extends REBIND to 4-ary sequences used in IDS-channel communication experiments.

Main components include:

- 4-ary edit-distance embedding
- IDS-specific adaptation
- residual whitening
- coordinate-wise soft-output calibration
- BCJR-guided REBIND V2 distillation

The communication setup uses a 204-dimensional output and a 4-ary alphabet `{0,1,2,3}` with padding handled separately.

### REBIND 4-ary V1 Inference

```bash
cd 4ary_setup

python scripts/eval/05_eval_detector.py \
  --data <ids_dataset.h5> \
  --checkpoint <rebind_v1_checkpoint.pt> \
  --whitener <whitener.pt> \
  --calibrator <calibrator.pt> \
  --output-dir outputs/v1
```

### REBIND 4-ary V2 Inference

```bash
cd 4ary_setup

python scripts/eval/06_eval_v2.py \
  --data <ids_dataset.h5> \
  --checkpoint <rebind_v2_checkpoint.pt> \
  --whitener <whitener.pt> \
  --calibrator <calibrator.pt> \
  --output-dir outputs/v2
```

BCJR is used as a teacher during V2 training only; it is not required for standard V2 inference.

## Notes

This repository is intended for research and reproducibility. Large datasets and trained checkpoints may be distributed separately.

## Citation

If you use this code in academic work, please cite the corresponding REBIND paper.
