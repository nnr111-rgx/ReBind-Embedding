# REBIND 4-ary setup

This folder is an isolated 4-ary IDS extension of the main `ReBind-Embedding` repository.

It is intentionally self-contained. The binary V1 code in the repository root does not need to be changed.

The pipeline is:

```text
generic 4-ary triplet pretraining
        ↓
IDS-specific fine-tuning on random 204-bit targets
        ↓
residual whitening
        ↓
whitening-aware coordinate training
        ↓
refit final whitener
        ↓
per-coordinate LLR calibration
        ↓
ReBind 4-ary V1 detector
        ↓
BCJR teacher generation
        ↓
BCJR-LLR distillation
        ↓
ReBind 4-ary V2 detector
```

## 1. Fixed 4-ary setup

The communication-facing setup uses:

```text
alphabet             {0,1,2,3}
PAD                  4
coded-bit dimension  204
embedding dimension  204
marker length T      142
marker period Np     7
IDS conditions       15
p_ins = p_del        {0.01, 0.02, 0.03}
p_sub                {0.01, 0.02, 0.03, 0.05, 0.10}
```

The marker encoder splits the 204 payload bits into two 102-bit streams and inserts the two binary marker patterns before mapping each pair to one 4-ary symbol:

```text
symbol = 2 * bit_stream_1 + bit_stream_2
0 -> 00
1 -> 01
2 -> 10
3 -> 11
```

The default `matlab` IDS mode follows the historical marker/forward-backward source version used by this folder. `standard` is also available as a separate channel mode for controlled ablations. Do not mix the two modes in one reported experiment.

## 2. Important reproducibility note

The high-level 4-ary protocol, marker setup, 15 IDS conditions, whitening path, LLR convention, and V2 distillation loss are reconstructed from the previous experiments.

The exact historical neural encoder source used to produce the old `rebind204_*` checkpoints was not fully recoverable. Therefore, the encoder in this folder is a clean self-contained ReBind implementation and the old numerical results must be treated as historical reference values, not as guaranteed expected outputs of this implementation.

The generic triplet generator in this folder also uses explicit repository defaults for positive and negative IDS severity. It is not claimed to reproduce an unrecovered historical triplet-distribution detail.

## 3. Installation

From the root repository:

```bash
conda activate ids-gpu
cd ~/NNR/ReBind-Embedding
pip install -e ./4ary_setup
```

Check the installation:

```bash
python - <<'PY'
import rebind4ary
print("T:", rebind4ary.T)
print("Np:", rebind4ary.NP)
print("N bits:", rebind4ary.N_BITS)
print("Embedding dim:", rebind4ary.EMBED_DIM)
PY
```

Expected:

```text
T: 142
Np: 7
N bits: 204
Embedding dim: 204
```

Run unit tests:

```bash
cd ~/NNR/ReBind-Embedding/4ary_setup
pytest -q
```

## 4. Directory layout

```text
4ary_setup/
├── README.md
├── pyproject.toml
├── requirements.txt
├── src/rebind4ary/
│   ├── bcjr.py
│   ├── calibration.py
│   ├── checkpoint.py
│   ├── config.py
│   ├── data.py
│   ├── edit_distance.py
│   ├── engine.py
│   ├── evaluation.py
│   ├── ids_channel.py
│   ├── losses.py
│   ├── marker.py
│   ├── metrics.py
│   ├── models.py
│   ├── utils.py
│   └── whitening.py
├── scripts/
│   ├── data/
│   │   ├── 01_generate_triplets.py
│   │   ├── 02_generate_ids_dataset.py
│   │   └── 03_generate_bcjr_teacher.py
│   ├── train/
│   │   ├── 01_train_triplet.py
│   │   ├── 02_finetune_ids.py
│   │   ├── 03_train_whitening_aware.py
│   │   └── 04_train_v2_distillation.py
│   ├── eval/
│   │   ├── 01_eval_triplet.py
│   │   ├── 02_eval_ids_geometry.py
│   │   ├── 03_fit_whitener.py
│   │   ├── 04_fit_llr_calibration.py
│   │   ├── 05_eval_detector.py
│   │   └── 06_eval_v2.py
│   └── plot/
│       ├── 01_plot_ed_scatter.py
│       ├── 02_plot_training_history.py
│       ├── 03_plot_condition_heatmap.py
│       └── 04_plot_llr_reliability.py
└── tests/
```

The commands below assume:

```bash
cd ~/NNR/ReBind-Embedding/4ary_setup
mkdir -p data runs logs figures
```

---

# Part A - Generic 4-ary edit-distance pretraining

## 5. Generate the generic 4-ary triplet dataset

The default dataset contains 20,000 triplets and is split internally into train/validation/test.

```bash
python scripts/data/01_generate_triplets.py \
  --output data/triplets_4ary_20k.h5 \
  --n-total 20000 \
  --seq-len 142 \
  --pos-p-ins 0.01 \
  --pos-p-del 0.01 \
  --pos-p-sub 0.01 \
  --neg-p-ins 0.03 \
  --neg-p-del 0.03 \
  --neg-p-sub 0.10 \
  --channel-mode matlab \
  --seed 4201 \
  2>&1 | tee logs/01_generate_triplets.log
```

The generator enforces:

```text
ED(anchor, positive) < ED(anchor, negative)
```

The exact AP, AN, and PN edit distances are stored in the H5 file.

## 6. Train the 204-dimensional triplet encoder

```bash
python scripts/train/01_train_triplet.py \
  --data data/triplets_4ary_20k.h5 \
  --save-dir runs/rebind204_triplet \
  --embed-dim 204 \
  --symbol-emb-dim 64 \
  --hidden-size 256 \
  --num-layers 2 \
  --dropout 0.1 \
  --epochs 100 \
  --batch-size 128 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --lambda-rank 0.2 \
  --rank-margin 0.2 \
  --patience 15 \
  --min-epochs 20 \
  --workers 4 \
  --seed 4301 \
  --device auto \
  2>&1 | tee logs/01_train_triplet.log
```

Main output:

```text
runs/rebind204_triplet/best.pt
```

## 7. Evaluate generic edit-distance geometry

Calibration is fitted on the training split only and then applied to test distances.

```bash
python scripts/eval/01_eval_triplet.py \
  --data data/triplets_4ary_20k.h5 \
  --checkpoint runs/rebind204_triplet/best.pt \
  --output-dir runs/rebind204_triplet_eval \
  --batch-size 256 \
  --workers 4 \
  --device auto
```

Outputs:

```text
runs/rebind204_triplet_eval/triplet_metrics.json
runs/rebind204_triplet_eval/ed_scatter.csv
```

Metrics include RMSE, MAE, Pearson, Spearman, and AP-vs-AN ranking accuracy.

Plot:

```bash
python scripts/plot/01_plot_ed_scatter.py \
  --csv runs/rebind204_triplet_eval/ed_scatter.csv \
  --output figures/triplet_ed_scatter.png
```

---

# Part B - IDS-specific 4-ary ReBind V1

## 8. Generate the fixed 15-condition IDS dataset

The default full protocol is:

```text
train       4000 frames / condition = 60,000
validation   800 frames / condition = 12,000
test         800 frames / condition = 12,000
```

Each frame starts from an independent random 204-bit vector. It is marker encoded to 142 4-ary symbols and passed through one of the 15 IDS conditions.

```bash
python scripts/data/02_generate_ids_dataset.py \
  --output data/ids_random204_15conditions.h5 \
  --train-per-condition 4000 \
  --val-per-condition 800 \
  --test-per-condition 800 \
  --channel-mode matlab \
  --l-max 2 \
  --seed 202607 \
  2>&1 | tee logs/02_generate_ids.log
```

The H5 stores the same information required for fair detector comparisons:

```text
sample_id
204 source/coded bits
clean marker-coded sequence
received noisy sequence
exact clean-to-noisy edit distance
condition_id
p_ins
p_del
p_sub
```

## 9. IDS-specific geometry fine-tuning

```bash
python scripts/train/02_finetune_ids.py \
  --data data/ids_random204_15conditions.h5 \
  --pretrained runs/rebind204_triplet/best.pt \
  --save-dir runs/rebind204_ids \
  --epochs 60 \
  --batch-size 128 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --distance-max 32 \
  --lambda-rank 0.2 \
  --lambda-preserve 0.05 \
  --patience 10 \
  --min-epochs 10 \
  --workers 4 \
  --seed 4302 \
  --device auto \
  2>&1 | tee logs/02_finetune_ids.log
```

Output:

```text
runs/rebind204_ids/best.pt
```

Evaluate the clean-to-IDS edit-distance geometry:

```bash
python scripts/eval/02_eval_ids_geometry.py \
  --data data/ids_random204_15conditions.h5 \
  --checkpoint runs/rebind204_ids/best.pt \
  --output-dir runs/rebind204_ids_eval \
  --split test \
  --batch-size 256 \
  --workers 4 \
  --device auto
```

## 10. Fit the initial residual whitener

Whitening is fitted only on TRAIN residuals:

```text
r = z_noisy - z_clean
```

```bash
python scripts/eval/03_fit_whitener.py \
  --data data/ids_random204_15conditions.h5 \
  --checkpoint runs/rebind204_ids/best.pt \
  --output runs/rebind204_whitening/whitener_init.pt \
  --split train \
  --batch-size 256 \
  --workers 4 \
  --shrinkage 0.01 \
  --eps-relative 1e-4 \
  --device auto \
  2>&1 | tee logs/03_fit_whitener_init.log
```

No validation or test residual is used to fit this transformation.

## 11. Whitening-aware coordinate training

This stage simultaneously encourages:

```text
coded-bit coordinate alignment
edit-distance preservation
edit-distance ranking
feature preservation
low residual coordinate correlation
```

```bash
python scripts/train/03_train_whitening_aware.py \
  --data data/ids_random204_15conditions.h5 \
  --pretrained runs/rebind204_ids/best.pt \
  --whitener runs/rebind204_whitening/whitener_init.pt \
  --save-dir runs/rebind204_ids_white \
  --epochs 100 \
  --batch-size 128 \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --distance-max 32 \
  --lambda-coordinate 1.0 \
  --lambda-ed 0.2 \
  --lambda-rank 0.2 \
  --lambda-preserve 0.05 \
  --lambda-whiteness 0.01 \
  --coordinate-temperature 1.0 \
  --patience 15 \
  --min-epochs 30 \
  --workers 4 \
  --seed 4303 \
  --device auto \
  2>&1 | tee logs/03_train_whitening_aware.log
```

Output:

```text
runs/rebind204_ids_white/best.pt
```

## 12. Refit the final whitener

Because the encoder changed during whitening-aware training, fit the reported final whitener from the final encoder using TRAIN residuals again.

```bash
python scripts/eval/03_fit_whitener.py \
  --data data/ids_random204_15conditions.h5 \
  --checkpoint runs/rebind204_ids_white/best.pt \
  --output runs/rebind204_whitening/whitener_final.pt \
  --split train \
  --batch-size 256 \
  --workers 4 \
  --shrinkage 0.01 \
  --eps-relative 1e-4 \
  --device auto \
  2>&1 | tee logs/04_fit_whitener_final.log
```

## 13. Fit the 204 per-coordinate LLR calibrators

The convention is:

```text
LLR_i = log P(bit_i = 0 | y) / P(bit_i = 1 | y)

LLR > 0 -> bit 0
LLR < 0 -> bit 1
```

Calibration is fit on TRAIN only.

```bash
python scripts/eval/04_fit_llr_calibration.py \
  --data data/ids_random204_15conditions.h5 \
  --checkpoint runs/rebind204_ids_white/best.pt \
  --whitener runs/rebind204_whitening/whitener_final.pt \
  --output runs/rebind204_llr/calibrator.pt \
  --split train \
  --batch-size 256 \
  --workers 4 \
  --newton-steps 30 \
  --device auto \
  2>&1 | tee logs/05_fit_llr_calibration.log
```

## 14. Final ReBind 4-ary V1 detector evaluation

```bash
python scripts/eval/05_eval_detector.py \
  --data data/ids_random204_15conditions.h5 \
  --checkpoint runs/rebind204_ids_white/best.pt \
  --whitener runs/rebind204_whitening/whitener_final.pt \
  --calibrator runs/rebind204_llr/calibrator.pt \
  --output-dir runs/rebind204_v1_test \
  --split test \
  --llr-clip 20 \
  --batch-size 256 \
  --workers 4 \
  --device auto
```

Outputs:

```text
runs/rebind204_v1_test/detector_metrics.json
runs/rebind204_v1_test/detector_metrics_by_condition.csv
runs/rebind204_v1_test/llr_outputs.npz
```

The NPZ keeps `sample_id`, `condition_id`, true bits, channel parameters, and the 204 LLRs so downstream LDPC/BP evaluation can reuse exactly the same frames.

---

# Part C - ReBind V2 with BCJR LLR distillation

## 15. Generate the oracle BCJR teacher targets

The teacher uses the same received sequences already stored in the IDS H5. It does not regenerate the channel.

It uses the true `p_ins`, `p_del`, and `p_sub` of each frame and emits 204 oracle marker-BCJR LLRs.

```bash
python scripts/data/03_generate_bcjr_teacher.py \
  --data data/ids_random204_15conditions.h5 \
  --output data/bcjr_teacher_random204.h5 \
  --splits train,validation,test \
  --workers 8 \
  --llr-clip 100 \
  2>&1 | tee logs/06_generate_bcjr_teacher.log
```

This is the slowest preprocessing step because the reference forward-backward decoder is CPU intensive. The output is generated once and then reused for all V2 experiments.

For V2 training, only TRAIN and validation teacher targets are required. Test teacher targets are optional unless the final evaluation also reports the oracle pre-BP detector on exactly the same test samples.

## 16. Train ReBind V2

V2 is initialized from the final ReBind 4-ary V1 encoder.

The default loss is:

```text
L_V2 =
    1.00 * L_soft_distill
  + 0.10 * L_LLR_Huber
  + 0.50 * L_hard_bit
  + 0.20 * L_ED
  + 0.05 * L_rank
  + 0.05 * L_preserve
  + 0.01 * L_whiteness
```

Default distillation settings:

```text
temperature = 2.0
learning rate = 1e-5
max epochs = 30
```

Validation student BCE is used for model selection.

```bash
python scripts/train/04_train_v2_distillation.py \
  --data data/ids_random204_15conditions.h5 \
  --teacher-data data/bcjr_teacher_random204.h5 \
  --v1-checkpoint runs/rebind204_ids_white/best.pt \
  --whitener runs/rebind204_whitening/whitener_final.pt \
  --calibrator runs/rebind204_llr/calibrator.pt \
  --save-dir runs/rebind204_v2 \
  --epochs 30 \
  --batch-size 128 \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --temperature 2.0 \
  --teacher-clip 20 \
  --distance-max 32 \
  --lambda-soft-distill 1.0 \
  --lambda-llr-huber 0.1 \
  --lambda-hard-bit 0.5 \
  --lambda-ed 0.2 \
  --lambda-rank 0.05 \
  --lambda-preserve 0.05 \
  --lambda-whiteness 0.01 \
  --patience 8 \
  --min-epochs 10 \
  --workers 4 \
  --seed 4304 \
  --device auto \
  2>&1 | tee logs/07_train_v2.log
```

Output:

```text
runs/rebind204_v2/best.pt
```

BCJR is used only to create teacher targets during training. It is not required at ReBind V2 inference.

## 17. Evaluate V2 on the fixed test set

With oracle teacher comparison:

```bash
python scripts/eval/06_eval_v2.py \
  --data data/ids_random204_15conditions.h5 \
  --teacher-data data/bcjr_teacher_random204.h5 \
  --checkpoint runs/rebind204_v2/best.pt \
  --whitener runs/rebind204_whitening/whitener_final.pt \
  --calibrator runs/rebind204_llr/calibrator.pt \
  --output-dir runs/rebind204_v2_test \
  --split test \
  --llr-clip 20 \
  --batch-size 256 \
  --workers 4 \
  --device auto
```

Without test teacher targets:

```bash
python scripts/eval/06_eval_v2.py \
  --data data/ids_random204_15conditions.h5 \
  --checkpoint runs/rebind204_v2/best.pt \
  --whitener runs/rebind204_whitening/whitener_final.pt \
  --calibrator runs/rebind204_llr/calibrator.pt \
  --output-dir runs/rebind204_v2_test \
  --split test \
  --llr-clip 20 \
  --batch-size 256 \
  --workers 4 \
  --device auto
```

Outputs:

```text
runs/rebind204_v2_test/v2_metrics.json
runs/rebind204_v2_test/v2_llr_outputs.npz
```

The main pre-BP metric is coded-bit BER from the student LLRs.

---

# Part D - Visualization

## 18. Training curves

Triplet training:

```bash
python scripts/plot/02_plot_training_history.py \
  --csv runs/rebind204_triplet/history.csv \
  --output figures/triplet_training.png \
  --y train_loss,val_loss
```

IDS fine-tuning:

```bash
python scripts/plot/02_plot_training_history.py \
  --csv runs/rebind204_ids/history.csv \
  --output figures/ids_training.png \
  --y train_loss,val_loss
```

Whitening-aware training:

```bash
python scripts/plot/02_plot_training_history.py \
  --csv runs/rebind204_ids_white/history.csv \
  --output figures/whitening_training.png \
  --y train_loss,val_loss,val_bit_acc
```

V2 distillation:

```bash
python scripts/plot/02_plot_training_history.py \
  --csv runs/rebind204_v2/history.csv \
  --output figures/v2_training.png \
  --y train_student_bce,val_student_bce,val_ber
```

## 19. BER by IDS condition

```bash
python scripts/plot/03_plot_condition_heatmap.py \
  --csv runs/rebind204_v1_test/detector_metrics_by_condition.csv \
  --metric ber \
  --output figures/v1_ber_by_condition.png
```

## 20. LLR reliability

V1:

```bash
python scripts/plot/04_plot_llr_reliability.py \
  --npz runs/rebind204_v1_test/llr_outputs.npz \
  --output figures/v1_llr_reliability.png
```

V2:

```bash
python scripts/plot/04_plot_llr_reliability.py \
  --npz runs/rebind204_v2_test/v2_llr_outputs.npz \
  --output figures/v2_llr_reliability.png
```

Every plotting script prints the final saved path after successful completion.

---

# Part E - Experimental rules

## 21. Train/validation/test separation

Use the splits as follows:

```text
TRAIN
- optimize encoder parameters
- fit residual whitener
- fit per-coordinate LLR calibrators
- generate BCJR targets used for V2 optimization

VALIDATION
- early stopping
- V2 checkpoint selection by student BCE
- hyperparameter decisions

TEST
- final geometry metrics
- final detector BER/calibration metrics
- final oracle-vs-V1-vs-V2 comparison
```

Do not fit whitening or LLR calibration on validation or test data.

## 22. Fair V1/V2 comparison

V1 and V2 must be evaluated on exactly the same test H5. Do not regenerate test channel realizations separately.

For every compared detector, retain and verify:

```text
sample_id
condition_id
true 204 bits
same received sequence
same test-frame count
same LLR convention
same LLR dimension [N, 204]
```

If downstream LDPC/BP decoding is added, the same H matrix, decoder implementation, schedule, iteration count, and LLR clipping must be used for all detector comparisons.

## 23. LLR convention

All detector-facing code in this folder uses:

```text
LLR = log P(bit=0 | y) / P(bit=1 | y)
```

Therefore:

```text
hard bit = 0 if LLR >= 0
hard bit = 1 if LLR < 0
```

Do not silently reverse this convention when connecting to an LDPC decoder.

## 24. Historical reference values

Previous experiments using the earlier 4-ary pipeline reported approximately:

```text
triplet pretraining
Pearson             0.8291
Spearman            0.8558
ranking accuracy    0.9591
calibrated MAE      3.627

IDS geometry fine-tuning
triplet-only P      0.6791
IDS fine-tuned P    0.7818
IDS fine-tuned MAE  2.8768

whitening-aware V1
raw Pearson         0.8848
raw Spearman        0.8903
raw MAE             2.1243
white noisy acc     0.8888
white clean acc     0.9909
mean offdiag corr   0.0104
condition number    2.72

per-coordinate LLR calibration
pre-BP BER          about 0.1035
BCE                  about 0.2943
ECE                  about 0.0212

BCJR-distilled V2 historical downstream run
pre-BP BER          0.08601
message BER         0.03653
BLER                0.26092
```

These values are historical references only. They are not acceptance thresholds for this cleaned implementation.

The classical oracle BCJR remains a separate baseline and should be reported independently. V2 should not be described as outperforming the oracle unless a directly matched experiment actually demonstrates it.

## 25. Recommended reporting order

For a paper or thesis, report:

```text
1. generic 4-ary ED geometry
2. IDS geometry before and after IDS fine-tuning
3. raw vs whitened residual statistics
4. V1 pre-BP BER and calibration
5. V2 pre-BP BER and calibration
6. condition-wise results over all 15 IDS settings
7. downstream LDPC/BP results using one shared decoder protocol
8. inference latency for V1, V2, and classical BCJR
```

Use multiple training seeds for the final neural-model result. A single seed is suitable for pipeline debugging, not the final statistical claim.
