# REBIND 

---

## 6. Stage 1 - Anchor reconstruction

Stage 1 trains the encoder, autoregressive decoder, and auxiliary anchor-bit head from scratch. Edit-distance and ranking losses are disabled.

```bash
python scripts/train/01_train_anchor_reconstruction.py \
  --h5 data/triplet_binary_sameIDS_K90_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --output-dir runs/base_D300_sameIDS_K90 \
  --embed-dim 300 \
  --symbol-emb-dim 16 \
  --hidden-size 256 \
  --num-layers 2 \
  --position-branch-dim 256 \
  --encoder-dropout 0.0 \
  --decoder-token-emb 64 \
  --decoder-context-dim 256 \
  --decoder-hidden 512 \
  --decoder-layers 2 \
  --decoder-dropout 0.1 \
  --batch-size 96 \
  --eval-batch-size 256 \
  --num-workers 2 \
  --lr-encoder 3e-4 \
  --lr-decoder 8e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --rank-margin 0.2 \
  --patience 20 \
  --min-epochs 20 \
  --epochs 80 \
  --seed 4100
```

Outputs:

```text
runs/base_D300_sameIDS_K90/stage1_anchor_reconstruction_best.pt
runs/base_D300_sameIDS_K90/stage1_anchor_reconstruction_history.csv
```

---

## 7. Stage 2 - Full reconstruction

Stage 2 initializes from the best Stage 1 checkpoint and trains reconstruction on anchor, positive, and negative sequences. ED and ranking losses remain disabled.

```bash
python scripts/train/02_train_full_reconstruction.py \
  --h5 data/triplet_binary_sameIDS_K90_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --output-dir runs/base_D300_sameIDS_K90 \
  --input-checkpoint runs/base_D300_sameIDS_K90/stage1_anchor_reconstruction_best.pt \
  --embed-dim 300 \
  --symbol-emb-dim 16 \
  --hidden-size 256 \
  --num-layers 2 \
  --position-branch-dim 256 \
  --encoder-dropout 0.0 \
  --decoder-token-emb 64 \
  --decoder-context-dim 256 \
  --decoder-hidden 512 \
  --decoder-layers 2 \
  --decoder-dropout 0.1 \
  --batch-size 96 \
  --eval-batch-size 256 \
  --num-workers 2 \
  --lr-encoder 2e-4 \
  --lr-decoder 5e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --rank-margin 0.2 \
  --patience 20 \
  --min-epochs 20 \
  --epochs 100 \
  --seed 4100
```

Outputs:

```text
runs/base_D300_sameIDS_K90/stage2_full_reconstruction_best.pt
runs/base_D300_sameIDS_K90/stage2_full_reconstruction_history.csv
```

---

## 8. Stage 3 - Edit-distance curriculum

Stage 3 keeps full reconstruction active while gradually introducing edit-distance regression and ranking.

```text
lambda_ED:   0.05 -> 0.50
lambda_rank: 0.02 -> 0.25
```

```bash
python scripts/train/03_train_ed_curriculum.py \
  --h5 data/triplet_binary_sameIDS_K90_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --output-dir runs/base_D300_sameIDS_K90 \
  --input-checkpoint runs/base_D300_sameIDS_K90/stage2_full_reconstruction_best.pt \
  --embed-dim 300 \
  --symbol-emb-dim 16 \
  --hidden-size 256 \
  --num-layers 2 \
  --position-branch-dim 256 \
  --encoder-dropout 0.0 \
  --decoder-token-emb 64 \
  --decoder-context-dim 256 \
  --decoder-hidden 512 \
  --decoder-layers 2 \
  --decoder-dropout 0.1 \
  --batch-size 96 \
  --eval-batch-size 256 \
  --num-workers 2 \
  --lr-encoder 1e-4 \
  --lr-decoder 3e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --rank-margin 0.2 \
  --patience 20 \
  --min-epochs 20 \
  --epochs 140 \
  --seed 4100
```

Outputs:

```text
runs/base_D300_sameIDS_K90/stage3_ed_curriculum_best.pt
runs/base_D300_sameIDS_K90/stage3_ed_curriculum_history.csv
```

---

## 9. Stage 4 - Final joint training

Stage 4 performs the final joint optimization with fixed ED and ranking weights.

```text
L = 1.00 * reconstruction
  + 0.25 * auxiliary bit reconstruction
  + 1.00 * edit-distance regression
  + 0.50 * ranking
```

```bash
python scripts/train/04_train_joint.py \
  --h5 data/triplet_binary_sameIDS_K90_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --output-dir runs/base_D300_sameIDS_K90 \
  --input-checkpoint runs/base_D300_sameIDS_K90/stage3_ed_curriculum_best.pt \
  --embed-dim 300 \
  --symbol-emb-dim 16 \
  --hidden-size 256 \
  --num-layers 2 \
  --position-branch-dim 256 \
  --encoder-dropout 0.0 \
  --decoder-token-emb 64 \
  --decoder-context-dim 256 \
  --decoder-hidden 512 \
  --decoder-layers 2 \
  --decoder-dropout 0.1 \
  --batch-size 96 \
  --eval-batch-size 256 \
  --num-workers 2 \
  --lr-encoder 5e-5 \
  --lr-decoder 1e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --rank-margin 0.2 \
  --patience 20 \
  --min-epochs 20 \
  --epochs 180 \
  --seed 4100
```

Outputs:

```text
runs/base_D300_sameIDS_K90/stage4_joint_best.pt
runs/base_D300_sameIDS_K90/stage4_joint_history.csv
runs/base_D300_sameIDS_K90/best.pt
```

`best.pt` is a copy of the best Stage 4 checkpoint and is the checkpoint used for the final V1 evaluation.

---

## 10. Training history

Each stage writes a CSV containing:

```text
epoch
train_loss
train_rec
train_bit
train_ed
train_rank
lambda_ed
lambda_rank
val_rec
val_bit
val_rmse
val_triplet
score
lr_encoder
lr_decoder
best
```

For Stage 1 and Stage 2, `val_rmse` and `val_triplet` are not active because edit-distance training is disabled.

The validation model-selection score is:

```text
Stage 1-2:
score = val_rec + lambda_bit * val_bit

Stage 3-4:
score = val_rec
      + lambda_bit * val_bit
      + 0.15 * val_rmse
      + 0.20 * (1 - val_triplet)
```

Early stopping begins only after `--min-epochs` and triggers after `--patience` consecutive non-improving epochs.

---

## 11. Reconstruction evaluation

Create the evaluation directory:

```bash
mkdir -p runs/base_D300_sameIDS_K90/eval
```

Run reconstruction-only evaluation:

```bash
python scripts/eval/01_eval_reconstruction.py \
  --h5 data/triplet_binary_sameIDS_K90_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --checkpoint runs/base_D300_sameIDS_K90/best.pt \
  --output-dir runs/base_D300_sameIDS_K90/eval \
  --batch-size 256 \
  --num-workers 2
```

Output:

```text
runs/base_D300_sameIDS_K90/eval/reconstruction_metrics.json
```

### Reconstruction metrics

| Metric | Meaning |
|---|---|
| `anchor_hamming_ar` | Main anchor Hamming error from autoregressive fixed-length decoding of the 100-bit anchor |
| `anchor_hamming_aux` | Hamming error from the auxiliary bit head; diagnostic only |
| `full_hamming_at_true_length` | Bit error over anchor, positive, and negative sequences when compared at their true lengths |
| `length_accuracy` | Fraction of reconstructed sequences with correct EOS-predicted length |
| `exact_reconstruction` | Fraction of sequences with both exact bits and exact length |
| `reconstruction_mean_edit_distance` | Mean exact edit distance between true and generated sequences |
| `reconstruction_normalized_edit_distance` | Mean edit distance normalized by true sequence length |

For the main reconstruction result, use:

```text
anchor_hamming_ar
```

Do **not** report `anchor_hamming_aux` as the main reconstruction result. The auxiliary head is a training/diagnostic head and bypasses the autoregressive decoder.

---

## 12. Edit-distance evaluation

Run edit-distance-only evaluation:

```bash
python scripts/eval/02_eval_edit_distance.py \
  --h5 data/triplet_binary_sameIDS_K90_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --checkpoint runs/base_D300_sameIDS_K90/best.pt \
  --output-dir runs/base_D300_sameIDS_K90/eval \
  --batch-size 256 \
  --num-workers 2
```

Outputs:

```text
runs/base_D300_sameIDS_K90/eval/edit_distance_metrics.json
runs/base_D300_sameIDS_K90/eval/ed_scatter.csv
```

### Edit-distance protocol

1. Encode the TRAIN split and compute raw Euclidean embedding distances.
2. Fit a single linear calibration on TRAIN only:

```text
estimated_ED = scale * raw_L2_distance + bias
```

3. Freeze that calibration.
4. Apply it to the untouched TEST split.
5. Compute the final test metrics.

The test split is never used to fit the calibration.

### Edit-distance metrics

| Metric | Meaning |
|---|---|
| `rmse` | RMSE between calibrated embedding distance and exact edit distance |
| `mae` | MAE between calibrated embedding distance and exact edit distance |
| `pearson` | Pearson correlation |
| `spearman` | Spearman rank correlation |
| `triplet` | Fraction of test triplets satisfying raw `d(A,P) < d(A,N)` in embedding space |

`ed_scatter.csv` contains:

```text
pair
true_ed
raw_distance
estimated_ed
```

where `pair` is one of `ap`, `an`, or `pn`.

---

## 13. Full final evaluation

The recommended final command evaluates both edit-distance approximation and reconstruction with the same final checkpoint.

```bash
python scripts/eval/03_eval_all.py \
  --h5 data/triplet_binary_sameIDS_K90_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --checkpoint runs/base_D300_sameIDS_K90/best.pt \
  --output-dir runs/base_D300_sameIDS_K90/eval \
  --batch-size 256 \
  --num-workers 2
```

Outputs:

```text
runs/base_D300_sameIDS_K90/eval/all_metrics.json
runs/base_D300_sameIDS_K90/eval/ed_scatter.csv
```

Inspect the result:

```bash
python -m json.tool runs/base_D300_sameIDS_K90/eval/all_metrics.json
```

The final JSON contains both metric groups:

```text
rmse
mae
pearson
spearman
triplet
anchor_hamming_ar
anchor_hamming_aux
full_hamming_at_true_length
length_accuracy
exact_reconstruction
reconstruction_mean_edit_distance
reconstruction_normalized_edit_distance
```

---

## 14. Visualizations

Create the figure directory:

```bash
mkdir -p runs/base_D300_sameIDS_K90/figures
```

### 14.1 Edit-distance scatter plot

```bash
python scripts/plot/01_plot_ed_scatter.py \
  --csv runs/base_D300_sameIDS_K90/eval/ed_scatter.csv \
  --output runs/base_D300_sameIDS_K90/figures/ed_scatter.png \
  --max-points 3000
```

Output:

```text
runs/base_D300_sameIDS_K90/figures/ed_scatter.png
```

The x-axis is exact edit distance and the y-axis is the train-calibrated estimated edit distance. The dashed diagonal is the ideal `estimated_ED = true_ED` line.

### 14.2 Stage 1 training history

```bash
python scripts/plot/02_plot_training_history.py \
  --csv runs/base_D300_sameIDS_K90/stage1_anchor_reconstruction_history.csv \
  --output runs/base_D300_sameIDS_K90/figures/stage1_history.png
```

### 14.3 Stage 2 training history

```bash
python scripts/plot/02_plot_training_history.py \
  --csv runs/base_D300_sameIDS_K90/stage2_full_reconstruction_history.csv \
  --output runs/base_D300_sameIDS_K90/figures/stage2_history.png
```

### 14.4 Stage 3 training history

```bash
python scripts/plot/02_plot_training_history.py \
  --csv runs/base_D300_sameIDS_K90/stage3_ed_curriculum_history.csv \
  --output runs/base_D300_sameIDS_K90/figures/stage3_history.png
```

### 14.5 Stage 4 training history

```bash
python scripts/plot/02_plot_training_history.py \
  --csv runs/base_D300_sameIDS_K90/stage4_joint_history.csv \
  --output runs/base_D300_sameIDS_K90/figures/stage4_history.png
```

The history plot shows:

```text
train_loss
validation reconstruction loss
validation ED RMSE, when active
```

Generate all four stage-history plots at once:

```bash
python scripts/plot/02_plot_training_history.py \
  --csv runs/base_D300_sameIDS_K90/stage1_anchor_reconstruction_history.csv \
  --output runs/base_D300_sameIDS_K90/figures/stage1_history.png

python scripts/plot/02_plot_training_history.py \
  --csv runs/base_D300_sameIDS_K90/stage2_full_reconstruction_history.csv \
  --output runs/base_D300_sameIDS_K90/figures/stage2_history.png

python scripts/plot/02_plot_training_history.py \
  --csv runs/base_D300_sameIDS_K90/stage3_ed_curriculum_history.csv \
  --output runs/base_D300_sameIDS_K90/figures/stage3_history.png

python scripts/plot/02_plot_training_history.py \
  --csv runs/base_D300_sameIDS_K90/stage4_joint_history.csv \
  --output runs/base_D300_sameIDS_K90/figures/stage4_history.png
```

After a complete main run, the important artifacts are:

```text
runs/base_D300_sameIDS_K90/
├── stage1_anchor_reconstruction_best.pt
├── stage1_anchor_reconstruction_history.csv
├── stage2_full_reconstruction_best.pt
├── stage2_full_reconstruction_history.csv
├── stage3_ed_curriculum_best.pt
├── stage3_ed_curriculum_history.csv
├── stage4_joint_best.pt
├── stage4_joint_history.csv
├── best.pt
├── eval/
│   ├── all_metrics.json
│   └── ed_scatter.csv
└── figures/
    ├── ed_scatter.png
    ├── stage1_history.png
    ├── stage2_history.png
    ├── stage3_history.png
    └── stage4_history.png
```

---

## 15. Optional embedding-dimension ablation

For a dimension ablation, train the same four-stage V1 pipeline independently for each dimension, for example:

```text
D = 64
D = 128
D = 300
```

Use separate directories:

```text
runs/ablation_dim/D64
runs/ablation_dim/D128
runs/ablation_dim/D300
```

Each directory must contain its own final evaluation:

```text
runs/ablation_dim/DXX/eval/all_metrics.json
```

After all runs are complete, collect them:

```bash
python scripts/ablation/01_collect_dimension.py \
  --root runs/ablation_dim \
  --dims 64 128 300 \
  --output runs/ablation_dim/summary.csv
```

Plot RMSE:

```bash
python scripts/plot/03_plot_ablation.py \
  --csv runs/ablation_dim/summary.csv \
  --x dimension \
  --metric rmse \
  --output runs/ablation_dim/rmse.png
```

Plot main autoregressive anchor Hamming error:

```bash
python scripts/plot/03_plot_ablation.py \
  --csv runs/ablation_dim/summary.csv \
  --x dimension \
  --metric anchor_hamming_ar \
  --output runs/ablation_dim/hamming.png
```

Plot triplet accuracy:

```bash
python scripts/plot/03_plot_ablation.py \
  --csv runs/ablation_dim/summary.csv \
  --x dimension \
  --metric triplet \
  --output runs/ablation_dim/triplet.png
```

The main historical V1 benchmark remains the D=300 SAME-IDS run described above. Do not mix results from different dataset-generation protocols in the same ablation table without clearly labeling the protocol difference.

---

## 16. Optional sequence-length ablation

Length ablation requires separate datasets and separate models because the position-preserving branch depends on the padded input length.

Typical lengths are:

```text
L = 50
L = 100
L = 200
```

The historical SAME-IDS L=100 dataset should remain the main benchmark. If L=50 or L=200 are generated with the repository's synthetic generator, they form a separate controlled ablation protocol and should not be described as the same historical dataset.

After independently training and evaluating each length, store:

```text
runs/ablation_length/L50/eval/all_metrics.json
runs/ablation_length/L100/eval/all_metrics.json
runs/ablation_length/L200/eval/all_metrics.json
```

Collect results:

```bash
python scripts/ablation/02_collect_length.py \
  --root runs/ablation_length \
  --lengths 50 100 200 \
  --output runs/ablation_length/summary.csv
```

Plot RMSE:

```bash
python scripts/plot/03_plot_ablation.py \
  --csv runs/ablation_length/summary.csv \
  --x length \
  --metric rmse \
  --output runs/ablation_length/rmse.png
```

Plot triplet accuracy:

```bash
python scripts/plot/03_plot_ablation.py \
  --csv runs/ablation_length/summary.csv \
  --x length \
  --metric triplet \
  --output runs/ablation_length/triplet.png
```

Plot autoregressive anchor Hamming error:

```bash
python scripts/plot/03_plot_ablation.py \
  --csv runs/ablation_length/summary.csv \
  --x length \
  --metric anchor_hamming_ar \
  --output runs/ablation_length/hamming.png
```

---

## 17. Optional synthetic triplet generator

The repository includes a generic exact-edit-distance binary triplet generator for controlled experiments.

Example:

```bash
python scripts/data/01_generate_triplets.py \
  --output data/triplet_binary_L100_N10000_seed1234_tiefree.h5 \
  --num-samples 10000 \
  --anchor-length 100 \
  --positive-min-ed 1 \
  --positive-max-ed 5 \
  --negative-min-ed 6 \
  --negative-max-ed 15 \
  --p-ins 0.3333333333 \
  --p-del 0.3333333333 \
  --p-sub 0.3333333333 \
  --seed 1234
```

Create an 80/10/10 split:

```bash
python scripts/data/02_make_split.py \
  --h5 data/triplet_binary_L100_N10000_seed1234_tiefree.h5 \
  --output splits/synthetic_split_L100.json \
  --train-frac 0.8 \
  --val-frac 0.1 \
  --seed 1234
```

Important: this generator uses a different triplet-generation protocol from the historical SAME-IDS K=90 dataset. Synthetic-generator results and historical-benchmark results must therefore be labeled separately.

---

## 18. Reproducibility and reporting rules

For the main V1 benchmark:

- Use the exact historical SAME-IDS H5.
- Use the exact historical common split.
- Fit final linear ED calibration on TRAIN only.
- Use validation data for model selection and early stopping only.
- Use TEST only for final metrics.
- Use `anchor_hamming_ar` as the main Hamming reconstruction metric.
- Keep `anchor_hamming_aux` as an auxiliary diagnostic.
- Do not compare Hamming metrics from different decoding protocols as if they were identical.
- Do not mix historical SAME-IDS results with generic synthetic-generator results without explicitly identifying the protocol difference.
- For paper-level mean and standard deviation, repeat the complete four-stage pipeline with multiple independent training seeds and aggregate the final TEST metrics. A single-seed run should be labeled as such.

The V1 checkpoint is tied to the padded length used to build its position branch. For the historical benchmark that length is `107`. Inputs requiring a larger padded length cannot be passed to the same checkpoint without changing the model architecture; this is important when designing later IDS fine-tuning experiments with insertions.

---

## 19. Quick end-to-end checklist

```text
[1] Install package
[2] Copy historical H5 and split
[3] Verify exact copies with cmp
[4] Verify N=10000, split=8000/1000/1000, max_len=107, anchor_len=100
[5] Train Stage 1
[6] Train Stage 2 from Stage 1 best
[7] Train Stage 3 from Stage 2 best
[8] Train Stage 4 from Stage 3 best
[9] Confirm runs/base_D300_sameIDS_K90/best.pt exists
[10] Run scripts/eval/03_eval_all.py
[11] Inspect all_metrics.json
[12] Generate ed_scatter.png
[13] Generate all four training-history plots
[14] Keep TEST untouched until final evaluation
```

---

## 20. Version roadmap

```text
V1
Clean REBIND base model
- reconstruction-aware embedding
- edit-distance regression
- ranking
- final clean benchmark
- visualization and ablation utilities

V2
IDS-channel fine-tuning
- initialize from pretrained V1
- insertion/deletion/substitution curriculum
- clean-reference reconstruction
- robustness evaluation across IDS levels

V3
Communication-system integration
- source/channel encoder
- IDS channel
- REBIND representation
- channel decoder
- BER / BLER / latency evaluation
```
