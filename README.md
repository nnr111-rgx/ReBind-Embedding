# REBIND v1

REBIND v1 is the base representation-learning project for reconstruction-aware edit-distance embedding over binary sequences.

This repository contains only the base REBIND stage. IDS fine-tuning and the communication-system stage are intended to be added later as separate versions.

## Architecture

```text
binary sequence
      |
      +------------------------------+
      |                              |
      v                              v
symbol embedding              signed bits + mask
      |                              |
2-layer BiGRU                       MLP
      |                              |
final hidden + masked attention      |
      |                              |
      +---------------+--------------+
                      |
                      v
                  projection
                      |
                   z in R^D
                  /       \
                 /         \
                v           v
        Euclidean ED    AR-GRU decoder
                         reconstruction
```

The auxiliary bit head is used during training to force the latent representation to retain exact bit information.

## Repository structure

```text
rebind-v1/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   └── rebind/
│       ├── __init__.py
│       ├── build.py
│       ├── checkpoint.py
│       ├── data.py
│       ├── edit_distance.py
│       ├── engine.py
│       ├── eval_cli.py
│       ├── evaluation.py
│       ├── losses.py
│       ├── metrics.py
│       ├── models.py
│       ├── reconstruction.py
│       └── train_cli.py
├── scripts/
│   ├── data/
│   │   ├── 01_generate_triplets.py
│   │   └── 02_make_split.py
│   ├── train/
│   │   ├── 01_train_anchor_reconstruction.py
│   │   ├── 02_train_full_reconstruction.py
│   │   ├── 03_train_ed_curriculum.py
│   │   └── 04_train_joint.py
│   ├── eval/
│   │   ├── 01_eval_reconstruction.py
│   │   ├── 02_eval_edit_distance.py
│   │   └── 03_eval_all.py
│   ├── ablation/
│   │   ├── 01_collect_dimension.py
│   │   └── 02_collect_length.py
│   └── plot/
│       ├── 01_plot_ed_scatter.py
│       ├── 02_plot_training_history.py
│       └── 03_plot_ablation.py
└── tests/
    └── test_models.py
```

## 1. Create the environment

```bash
conda create -n rebind python=3.11 -y
conda activate rebind
pip install -r requirements.txt
pip install -e .
```

Test the installation:

```bash
python - <<'PY'
import rebind
print("REBIND import OK")
PY
```

## 2. Generate the base binary edit-distance dataset

The default generator creates 10,000 binary triplets with clean anchor length 100.

Positive samples are generated at exact edit distances 1 to 5.

Negative samples are generated at exact edit distances 6 to 15.

Insertion, deletion, and substitution use the same operation probabilities.

The triplets are tie-free.

```bash
mkdir -p data
```

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

The generated HDF5 format is:

```text
sample_id/
├── anchor
├── positive
└── negative
```

Each group contains:

```text
d_anchor_positive
d_anchor_negative
d_positive_negative
```

## 3. Create the common 80/10/10 split

```bash
mkdir -p splits
```

```bash
python scripts/data/02_make_split.py \
  --h5 data/triplet_binary_L100_N10000_seed1234_tiefree.h5 \
  --output splits/common_split_L100.json \
  --train-frac 0.8 \
  --val-frac 0.1 \
  --seed 1234
```

Expected size:

```text
train = 8000
val   = 1000
test  = 1000
```

## 4. Stage 1: anchor reconstruction

This stage trains the encoder, autoregressive GRU decoder, and auxiliary bit head from scratch.

Edit-distance loss is disabled.

```bash
mkdir -p runs/base_D300
```

```bash
python scripts/train/01_train_anchor_reconstruction.py \
  --h5 data/triplet_binary_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --output-dir runs/base_D300 \
  --embed-dim 300 \
  --lr-encoder 3e-4 \
  --lr-decoder 8e-4 \
  --epochs 80 \
  --batch-size 96 \
  --seed 4100
```

Output:

```text
runs/base_D300/stage1_anchor_reconstruction_best.pt
runs/base_D300/stage1_anchor_reconstruction_history.csv
```

## 5. Stage 2: full reconstruction

This stage loads the Stage 1 checkpoint and trains reconstruction on anchor, positive, and negative sequences.

Edit-distance loss is still disabled.

```bash
python scripts/train/02_train_full_reconstruction.py \
  --h5 data/triplet_binary_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --output-dir runs/base_D300 \
  --input-checkpoint runs/base_D300/stage1_anchor_reconstruction_best.pt \
  --embed-dim 300 \
  --lr-encoder 2e-4 \
  --lr-decoder 5e-4 \
  --epochs 100 \
  --batch-size 96 \
  --seed 4100
```

Output:

```text
runs/base_D300/stage2_full_reconstruction_best.pt
runs/base_D300/stage2_full_reconstruction_history.csv
```

## 6. Stage 3: edit-distance curriculum

This stage keeps reconstruction active while gradually introducing edit-distance regression and ranking.

The ED regression weight is increased from 0.05 to 0.50.

The ranking weight is increased from 0.02 to 0.25.

```bash
python scripts/train/03_train_ed_curriculum.py \
  --h5 data/triplet_binary_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --output-dir runs/base_D300 \
  --input-checkpoint runs/base_D300/stage2_full_reconstruction_best.pt \
  --embed-dim 300 \
  --lr-encoder 1e-4 \
  --lr-decoder 3e-4 \
  --epochs 140 \
  --batch-size 96 \
  --seed 4100
```

Output:

```text
runs/base_D300/stage3_ed_curriculum_best.pt
runs/base_D300/stage3_ed_curriculum_history.csv
```

## 7. Stage 4: final joint training

The final objective is:

```text
L =
1.00 reconstruction
+ 0.25 auxiliary bit reconstruction
+ 1.00 edit-distance regression
+ 0.50 ranking
```

```bash
python scripts/train/04_train_joint.py \
  --h5 data/triplet_binary_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --output-dir runs/base_D300 \
  --input-checkpoint runs/base_D300/stage3_ed_curriculum_best.pt \
  --embed-dim 300 \
  --lr-encoder 5e-5 \
  --lr-decoder 1e-4 \
  --epochs 180 \
  --batch-size 96 \
  --seed 4100
```

Final checkpoint:

```text
runs/base_D300/best.pt
```

## 8. Reconstruction evaluation

```bash
mkdir -p runs/base_D300/eval
```

```bash
python scripts/eval/01_eval_reconstruction.py \
  --h5 data/triplet_binary_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --checkpoint runs/base_D300/best.pt \
  --output-dir runs/base_D300/eval
```

Main reconstruction metrics:

```text
anchor_hamming_ar
anchor_hamming_aux
full_hamming_at_true_length
length_accuracy
exact_reconstruction
reconstruction_mean_edit_distance
reconstruction_normalized_edit_distance
```

The main reconstruction Hamming metric is:

```text
anchor_hamming_ar
```

## 9. Edit-distance evaluation

The final edit-distance evaluation fits one linear calibration on TRAIN raw Euclidean distances and evaluates the untouched TEST split.

```bash
python scripts/eval/02_eval_edit_distance.py \
  --h5 data/triplet_binary_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --checkpoint runs/base_D300/best.pt \
  --output-dir runs/base_D300/eval
```

Outputs:

```text
runs/base_D300/eval/edit_distance_metrics.json
runs/base_D300/eval/ed_scatter.csv
```

Metrics:

```text
RMSE
MAE
Pearson
Spearman
Triplet accuracy
```

## 10. Full evaluation

```bash
python scripts/eval/03_eval_all.py \
  --h5 data/triplet_binary_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --checkpoint runs/base_D300/best.pt \
  --output-dir runs/base_D300/eval
```

Output:

```text
runs/base_D300/eval/all_metrics.json
runs/base_D300/eval/ed_scatter.csv
```

## 11. Edit-distance scatter plot

```bash
mkdir -p runs/base_D300/figures
```

```bash
python scripts/plot/01_plot_ed_scatter.py \
  --csv runs/base_D300/eval/ed_scatter.csv \
  --output runs/base_D300/figures/ed_scatter.png
```

## 12. Training-history plots

Stage 1:

```bash
python scripts/plot/02_plot_training_history.py \
  --csv runs/base_D300/stage1_anchor_reconstruction_history.csv \
  --output runs/base_D300/figures/stage1_history.png
```

Stage 2:

```bash
python scripts/plot/02_plot_training_history.py \
  --csv runs/base_D300/stage2_full_reconstruction_history.csv \
  --output runs/base_D300/figures/stage2_history.png
```

Stage 3:

```bash
python scripts/plot/02_plot_training_history.py \
  --csv runs/base_D300/stage3_ed_curriculum_history.csv \
  --output runs/base_D300/figures/stage3_history.png
```

Stage 4:

```bash
python scripts/plot/02_plot_training_history.py \
  --csv runs/base_D300/stage4_joint_history.csv \
  --output runs/base_D300/figures/stage4_history.png
```

## 13. Dimension ablation

Run the full four-stage pipeline separately for each embedding dimension.

### D = 64

```bash
python scripts/train/01_train_anchor_reconstruction.py \
  --h5 data/triplet_binary_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --output-dir runs/ablation_dim/D64 \
  --embed-dim 64 \
  --lr-encoder 3e-4 \
  --lr-decoder 8e-4 \
  --epochs 80
```

```bash
python scripts/train/02_train_full_reconstruction.py \
  --h5 data/triplet_binary_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --output-dir runs/ablation_dim/D64 \
  --input-checkpoint runs/ablation_dim/D64/stage1_anchor_reconstruction_best.pt \
  --embed-dim 64 \
  --lr-encoder 2e-4 \
  --lr-decoder 5e-4 \
  --epochs 100
```

```bash
python scripts/train/03_train_ed_curriculum.py \
  --h5 data/triplet_binary_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --output-dir runs/ablation_dim/D64 \
  --input-checkpoint runs/ablation_dim/D64/stage2_full_reconstruction_best.pt \
  --embed-dim 64 \
  --lr-encoder 1e-4 \
  --lr-decoder 3e-4 \
  --epochs 140
```

```bash
python scripts/train/04_train_joint.py \
  --h5 data/triplet_binary_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --output-dir runs/ablation_dim/D64 \
  --input-checkpoint runs/ablation_dim/D64/stage3_ed_curriculum_best.pt \
  --embed-dim 64 \
  --lr-encoder 5e-5 \
  --lr-decoder 1e-4 \
  --epochs 180
```

```bash
python scripts/eval/03_eval_all.py \
  --h5 data/triplet_binary_L100_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L100.json \
  --checkpoint runs/ablation_dim/D64/best.pt \
  --output-dir runs/ablation_dim/D64/eval
```

### D = 128

Use the same five commands and replace:

```text
D64
```

with:

```text
D128
```

and:

```text
--embed-dim 64
```

with:

```text
--embed-dim 128
```

### D = 300

Use the same five commands and replace:

```text
D64
```

with:

```text
D300
```

and:

```text
--embed-dim 64
```

with:

```text
--embed-dim 300
```

Collect dimension ablation:

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

Plot Hamming:

```bash
python scripts/plot/03_plot_ablation.py \
  --csv runs/ablation_dim/summary.csv \
  --x dimension \
  --metric anchor_hamming_ar \
  --output runs/ablation_dim/hamming.png
```

## 14. Sequence-length ablation

Generate a separate dataset for each sequence length.

### L = 50

```bash
python scripts/data/01_generate_triplets.py \
  --output data/triplet_binary_L50_N10000_seed1234_tiefree.h5 \
  --num-samples 10000 \
  --anchor-length 50 \
  --positive-min-ed 1 \
  --positive-max-ed 5 \
  --negative-min-ed 6 \
  --negative-max-ed 15 \
  --seed 1234
```

```bash
python scripts/data/02_make_split.py \
  --h5 data/triplet_binary_L50_N10000_seed1234_tiefree.h5 \
  --output splits/common_split_L50.json \
  --seed 1234
```

Run the same four training stages with:

```text
data/triplet_binary_L50_N10000_seed1234_tiefree.h5
splits/common_split_L50.json
runs/ablation_length/L50
```

Then evaluate:

```bash
python scripts/eval/03_eval_all.py \
  --h5 data/triplet_binary_L50_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L50.json \
  --checkpoint runs/ablation_length/L50/best.pt \
  --output-dir runs/ablation_length/L50/eval
```

### L = 100

Use:

```text
data/triplet_binary_L100_N10000_seed1234_tiefree.h5
splits/common_split_L100.json
runs/ablation_length/L100
```

### L = 200

Generate:

```bash
python scripts/data/01_generate_triplets.py \
  --output data/triplet_binary_L200_N10000_seed1234_tiefree.h5 \
  --num-samples 10000 \
  --anchor-length 200 \
  --positive-min-ed 1 \
  --positive-max-ed 5 \
  --negative-min-ed 6 \
  --negative-max-ed 15 \
  --seed 1234
```

Create the split:

```bash
python scripts/data/02_make_split.py \
  --h5 data/triplet_binary_L200_N10000_seed1234_tiefree.h5 \
  --output splits/common_split_L200.json \
  --seed 1234
```

Run the same four training stages with:

```text
data/triplet_binary_L200_N10000_seed1234_tiefree.h5
splits/common_split_L200.json
runs/ablation_length/L200
```

Evaluate:

```bash
python scripts/eval/03_eval_all.py \
  --h5 data/triplet_binary_L200_N10000_seed1234_tiefree.h5 \
  --split-json splits/common_split_L200.json \
  --checkpoint runs/ablation_length/L200/best.pt \
  --output-dir runs/ablation_length/L200/eval
```

Collect length ablation:

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

Plot Triplet:

```bash
python scripts/plot/03_plot_ablation.py \
  --csv runs/ablation_length/summary.csv \
  --x length \
  --metric triplet \
  --output runs/ablation_length/triplet.png
```

## 15. Tests

Install pytest:

```bash
pip install pytest
```

Run:

```bash
pytest -q
```

## 16. Recommended GitHub upload

Initialize the repository:

```bash
git init
git branch -M main
```

Check the files:

```bash
git status
```

Add:

```bash
git add .
```

Commit:

```bash
git commit -m "Initial REBIND v1 implementation"
```

Create an empty GitHub repository named:

```text
rebind
```

Then connect it:

```bash
git remote add origin https://github.com/nnr111-rgx/rebind.git
```

Push:

```bash
git push -u origin main
```

## 17. Planned next versions

```text
v1
Base REBIND
sequence reconstruction
edit-distance embedding
ablation
evaluation
plots

v2
IDS fine-tuning
pretrained REBIND
insertion/deletion/substitution curriculum
robust latent representation

v3
Communication setup
source encoder
IDS channel
REBIND representation
channel decoder
BER
BLER
latency
```
