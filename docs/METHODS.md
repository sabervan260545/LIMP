# LIMP methods and implementation boundaries

## LIMP-DI

LIMP-DI models AMP/non-AMP discrimination and uses gradients from the
discriminative objective to guide inverse sequence design. The corrected
evaluation uses cluster-disjoint positive/negative partitions, reports AUROC,
AUPRC, MCC, balanced accuracy and calibration, and includes a simple
physicochemical logistic baseline and shuffled-sequence sensitivity. Across
five seeds, corrected LIMP-DI achieved mean AUROC 0.9914 and AUPRC 0.9888; these
values describe discrimination under the frozen computational split, not
experimental AMP activity.

The legacy and corrected implementations are both retained so that padding,
readout and regularization changes remain auditable. LIMP-DI-generated bridge
samples were used only to compare the two method formulations under common
output constraints.

## LIMP-AR

For tokens `x_1, ..., x_L`, teacher-forced training minimizes the masked
negative log likelihood

```text
L_NLL = - sum_t 1[x_t != PAD] log p_theta(x_t | x_<t).
```

The model is a compact pre-norm causal Transformer. For each layer, a schematic
update is

```text
u = h + CausalAttention(LayerNorm(h))
h_next = u + FFN(LayerNorm(u)).
```

A final LayerNorm and output linear layer produce vocabulary logits. At sampling
step t, logits are temperature-scaled and truncated before softmax:

```text
z_t' = TopK/TopP(z_t / temperature)
p_t = softmax(z_t').
```

The runtime rejects invalid or duplicate sequences and refills with new seeded
batches. This separates sequence modelling from the exact-N delivery contract.

## Training split

Canonicalized unique AMP sequences are grouped by connected components under
same-length ungapped identity >= 0.80. Whole clusters, rather than individual
sequences, are assigned to train/validation/test using seed 42. The release
distributes only SHA-256 sequence identifiers, lengths, source classes, cluster
IDs and split labels.

## Relationship to AMP-Agent

LIMP-AR returns sequences only. AMP-Agent separately performs ESMFold structure
prediction, PGAT-ABPp qualification, organism-specific MIC prediction,
hemolysis and CPP evaluation, structural descriptor QC, six-objective
epsilon-Pareto analysis and deterministic HBSP ranking. No downstream
predictor score is used during LIMP-AR training or token sampling.
