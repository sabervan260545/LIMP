# LIMP: discriminative inverse design and scalable autoregressive AMP generation

LIMP is a laboratory method family for antimicrobial-peptide sequence design.
This release keeps two implementations scientifically and operationally
separate:

- **LIMP-DI** is the original discriminative inverse-design formulation. It
  studies whether AMP/non-AMP discrimination can guide peptide design.
- **LIMP-AR** is a production-oriented autoregressive formulation trained on
  AMP sequences. It generates reproducible, fixed-size candidate pools and is
  the only LIMP model used to create the 100,000-sequence pool evaluated by the
  AMP-Agent full-structure study.

They are successive members of the same method family, not the same model.
Historical code and service identifiers retain `LIUP`/`liup_generator` so that
frozen checkpoints and run manifests remain traceable. See
[`docs/NAME_MAPPING.md`](docs/NAME_MAPPING.md).

## Release scope

This repository contains:

- executable LIMP-DI and LIMP-AR source code;
- the formal LIMP-AR checkpoint;
- exact-N sampling and sequence-only API code;
- a sequence-hash-only split manifest, with no raw training FASTA;
- compact DI validation, AR multi-seed, DI-versus-AR bridge, diversity and
  formal 100k traceability evidence;
- unit tests, CI configuration and a deterministic release verifier.

It does **not** contain MIC, hemolysis, CPP, ESMFold, PGAT-ABPp or HBSP code and
does not perform biological candidate ranking. Those operations belong to the
AMP-Agent downstream pipeline. The two 100,000-sequence candidate pools and
large candidate-level evidence remain in the separate Zenodo-oriented data
archive rather than this Git repository.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
python scripts/verify_public_release.py
```

Start the sequence-only API:

```bash
LIMP_MODEL_PATH=checkpoint/LIMP-AR_generator.pt \
LIMP_DEVICE=cpu \
uvicorn app:app --host 127.0.0.1 --port 8011
```

Example request:

```bash
curl -s http://127.0.0.1:8011/generate \
  -H 'content-type: application/json' \
  -d '{"n":10,"min_length":12,"max_length":28,"temperature":1.0,"top_k":10,"top_p":1.0,"seed":20260727}'
```

The formal 100k parameters are frozen in
[`configs/formal_100k_generation_config.json`](configs/formal_100k_generation_config.json).
The HTTP endpoint accepts at most 2,048 sequences per request; AMP-Agent
obtained exact N by persistent batched requests, validity checks,
deduplication and deterministic refill.

## Scientific boundaries

- LIMP-AR proposes sequences; it does not predict activity or safety.
- Model log probability is not an MIC, AMP probability or ranking score.
- The formal structural study used LIMP-AR only. LIMP-DI results establish the
  discriminative inverse-design path and are not substituted for the 100k
  generator.
- All reported biological and structural properties are computational and
  require experimental validation.
- Raw training sequences are withheld until source-level redistribution rights
  are documented. The release provides hashes, cluster IDs and splits instead.

See [`MODEL_CARD.md`](MODEL_CARD.md), [`docs/METHODS.md`](docs/METHODS.md) and
[`RELEASE_ASSETS.md`](RELEASE_ASSETS.md) for full details.

## Release status

The code, checkpoint, evidence, manifests and MIT license are technically
verified for an author-controlled Zenodo upload. The citation record uses the
neutral author label `LIMP contributors`; replace it with the laboratory's
approved author list and add the public repository URL when those details are
available. Raw training FASTA remains withheld by design.
