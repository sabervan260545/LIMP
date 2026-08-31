# Verification report

The standalone release candidate passed the following local gates on
2026-08-31:

- six Python tests passed;
- checkpoint SHA-256, architecture and 407,703-parameter count verified;
- 5,652 hash-only split records verified with no sequence column;
- formal 100k seed, length and sampling parameters verified;
- no raw FASTA files present;
- no credentials, private IP addresses, usernames or internal absolute server
  paths detected in distributed text files;
- all release files covered by `SHA256SUMS.txt`.

Technical release readiness and author-controlled public upload readiness are
true under the MIT license included in this package. The neutral citation label
`LIMP contributors` and the absent public repository URL should be replaced by
the laboratory's approved metadata when available; this does not affect the
checkpoint or computational evidence. Supplementary Information is maintained
in the separate AMP-Agent publication package.
