# Model card: LIMP-DI and LIMP-AR

## Model family

LIMP contains two explicitly separated models.

| Model | Learning problem | Scientific role | Used for formal 100k pool |
|---|---|---|---:|
| LIMP-DI | AMP/non-AMP discrimination plus gradient-guided inverse design | Establish discriminative design feasibility | No |
| LIMP-AR | Causal language modelling on AMP sequences | Scalable exact-size candidate proposal | Yes |

The checkpoint in this repository is **LIMP-AR**. LIMP-DI code and compact
validation evidence are retained for method continuity, but a historical DI
checkpoint is not presented as the source of the structural-study candidates.

## LIMP-AR architecture

LIMP-AR factorizes a peptide sequence as

```text
p(x_1, ..., x_L) = product_t p(x_t | x_<t).
```

Input tokens comprise `<PAD>`, `<BOS>`, `<EOS>` and the 20 canonical amino
acids. Token and learnable positional embeddings are summed and passed through
three pre-normalized causal Transformer blocks. Each block contains causal
multi-head self-attention and a feed-forward network, each with a residual
connection. A final LayerNorm and linear vocabulary projection produce
next-residue logits.

| Parameter | Frozen value |
|---|---:|
| Parameters | 407,703 |
| Maximum tokens | 32 |
| Hidden dimension | 128 |
| Transformer layers | 3 |
| Attention heads | 4 |
| Feed-forward dimension | 256 |
| Dropout | 0.15 |
| Activation | GELU |
| Normalization | pre-norm, plus final LayerNorm |

Sampling applies temperature scaling and optional top-k/top-p truncation before
multinomial residue selection. `<EOS>` is disabled before the requested minimum
length and forced at the maximum length. Exact-N generation is a runtime
property: invalid/duplicate outputs are rejected and the persistent caller
requests deterministic refill batches until N unique sequences are accepted.

## Training data and split

The frozen input contained 5,652 unique AMP sequences with lengths 4–30. The
hash-only manifest reports 4,331 training, 613 validation and 708 test records.
Same-length sequence identity at or above 0.80 was clustered before splitting,
preventing those cluster neighbours from crossing partitions. Raw sequences are
not distributed in this repository.

## Training

| Setting | Value |
|---|---:|
| Seed | 42 |
| Epochs | 200 |
| Batch size | 256 |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Label smoothing | 0.05 |
| Best validation NLL | 2.24984 |
| Test NLL | 2.19248 |
| Test perplexity | 8.95737 |

The distributed checkpoint SHA-256 is
`c25b229bd85f321926d83c0976d4c13372bff6f23e5f645a6535a1f6074562d8`.

## Stability and formal generation

Three 10,000-sequence runs using seeds 20260727–20260729 each reached exact
count, 100% token validity and 100% uniqueness after refill. Raw duplicate rates
were 0.210–0.259%. The formal run used seed 20260727, temperature 1.0, top-k 10,
top-p 1.0 and lengths 12–28; it returned 100,000 unique sequences from 100,452
raw outputs in 50 persistent batches.

The 100k pool then entered the external AMP-Agent pipeline. ESMFold succeeded
for all 100,000 sequences, 98,690 passed PGAT-ABPp, 98,674 also passed the
pLDDT qualification gate, and 33,180 passed frozen pre-Pareto QC and entered
formal HBSP ranking. These downstream counts are evidence about the complete
pipeline, not intrinsic LIMP-AR accuracy claims.

## Intended use

- research candidate generation for short canonical AMP sequences;
- controlled comparison of candidate generators;
- upstream proposal generation before independent activity, safety and
  structure evaluation.

## Out-of-scope use and limitations

- no clinical, therapeutic or biosafety decision making;
- no claim of experimental antimicrobial activity;
- no noncanonical residues or post-translational modifications;
- no species-specific conditioning inside LIMP-AR;
- no guarantee of novelty, selectivity, low toxicity or protease stability;
- no use of the LIMP-DI classifier score as a downstream HBSP objective.

The training corpus provenance is recorded at hash and source-class level, but
raw-record redistribution remains pending. The model can reproduce or closely
approximate training sequences; candidate-level novelty must therefore be
audited before synthesis.
