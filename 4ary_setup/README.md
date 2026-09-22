# REBIND 4-ary IDS Fine-Tuning

This folder fine-tunes the pretrained REBIND embedding for the 4-ary IDS communication setup before any downstream communication decoder is introduced.

The REBIND encoder architecture is not replaced. The code imports `REBINDEncoder` directly from the root `rebind` package and initializes it from an existing REBIND checkpoint.

## Data path

```text
100 information bits
→ terminated convolutional code, K=3, generators (5,7) octal
→ 204 coded bits
→ marker encoder, T=142, Np=7
→ 142 clean 4-ary symbols
→ IDS corruption
```

A 4-ary symbol is converted internally to its two-bit representation before it is passed to the pretrained binary REBIND encoder. This keeps the original REBIND symbol embedding and recurrent architecture intact.

The only shape adaptation is the first position-branch layer because communication sequences are longer than the original binary REBIND input. Existing position weights are copied to their corresponding old positions and all newly introduced positions start at zero.

## IDS curriculum

- Stage 1: `p_ins=p_del=0.01`, `p_sub∈[0.01,0.05]`
- Stage 2: `p_ins=p_del=0.02`, `p_sub∈[0.01,0.05]`
- Stage 3: `p_ins=p_del=0.03`, `p_sub∈[0.01,0.05]`

Stage 1 starts from the pretrained REBIND checkpoint. Stage 2 starts from Stage 1. Stage 3 starts from Stage 2.

Fine-tuning uses clean/noisy contrastive supervision plus an auxiliary 204-bit head. The auxiliary head is based on the same MLP structure as REBIND's `AnchorBitHead`; it is not a replacement for the 300-D embedding.

## Evaluation

The Stage-3 embedding is evaluated on 30 fresh IDS conditions:

- `p_ins=p_del ∈ {0.01,0.02,0.03}`
- `p_sub ∈ {0.01,...,0.10}`

Reported metrics are clean/noisy cosine similarity, embedding L2 drift, clean/noisy coded-bit BER from the auxiliary head, and batch retrieval top-1 accuracy.

Communication-specific models should only be added after this embedding evaluation is satisfactory.
