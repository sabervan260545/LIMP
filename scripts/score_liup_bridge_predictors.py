#!/usr/bin/env python3
"""Score frozen bridge pools with Macrel and HydrAMP using batch inference."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def collect(bridge_dir: Path, ar_dir: Path, seeds: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for implementation, path_template in (
        ("LIMP-DI-corrected", bridge_dir / "liup_di_corrected/seed_{seed}/sequences.csv"),
        ("LIMP-AR", ar_dir / "seed_{seed}/sequences.csv"),
    ):
        for seed in seeds:
            path = Path(str(path_template).format(seed=seed))
            frame = pd.read_csv(path, usecols=["sequence_sha256", "sequence", "length"])
            frame.insert(0, "implementation", implementation)
            frame.insert(1, "seed", seed)
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.insert(0, "predictor_id", [f"bridge_{index:06d}" for index in range(len(combined))])
    if combined["predictor_id"].duplicated().any() or len(combined) != 2 * len(seeds) * 10000:
        raise AssertionError("Unexpected bridge pool cardinality")
    return combined


def write_fasta(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in frame.itertuples(index=False):
            handle.write(f">{row.predictor_id}\n{row.sequence}\n")


def run_macrel(frame: pd.DataFrame, input_fasta: Path, output_dir: Path, container: str) -> pd.DataFrame:
    remote_fasta = "/tmp/liup_bridge_predictors.fasta"
    remote_output = "/tmp/liup_bridge_macrel"
    run(["docker", "cp", str(input_fasta), f"{container}:{remote_fasta}"])
    subprocess.run(["docker", "exec", container, "rm", "-rf", remote_output], check=False)
    run(["docker", "exec", "-e", "OMP_NUM_THREADS=8", container, "macrel", "peptides", "--fasta", remote_fasta, "--output", remote_output, "--threads", "8", "--keep-negatives", "--force"])
    local_gzip = output_dir / "macrel.out.prediction.gz"
    run(["docker", "cp", f"{container}:{remote_output}/macrel.out.prediction.gz", str(local_gzip)])
    macrel = pd.read_csv(local_gzip, sep="\t", comment="#")
    identifier_column = next((column for column in ("Access", "access", "ID", "id", "Name", "name") if column in macrel.columns), None)
    if identifier_column is None or "AMP_probability" not in macrel.columns:
        raise ValueError(f"Unexpected Macrel columns: {macrel.columns.tolist()}")
    mapped = frame[["predictor_id"]].merge(macrel[[identifier_column, "AMP_probability"]].rename(columns={identifier_column: "predictor_id", "AMP_probability": "macrel_probability"}), on="predictor_id", how="left", validate="one_to_one")
    return mapped


def run_hydramp(frame: pd.DataFrame, output_dir: Path, container: str, script: Path) -> pd.DataFrame:
    input_csv = output_dir / "hydramp_input.csv"
    output_csv = output_dir / "hydramp_scores.csv"
    frame[["predictor_id", "sequence"]].to_csv(input_csv, index=False)
    run(["docker", "cp", str(input_csv), f"{container}:/tmp/liup_bridge_hydramp_input.csv"])
    run(["docker", "cp", str(script), f"{container}:/tmp/score_amp_classifier_batch.py"])
    run(["docker", "exec", container, "python", "/tmp/score_amp_classifier_batch.py", "--input-csv", "/tmp/liup_bridge_hydramp_input.csv", "--output-csv", "/tmp/liup_bridge_hydramp_scores.csv", "--model-path", "/app/model_weights/37"])
    run(["docker", "cp", f"{container}:/tmp/liup_bridge_hydramp_scores.csv", str(output_csv)])
    return pd.read_csv(output_csv)


def directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file.relative_to(path)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(file)))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--ar-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260727, 20260728, 20260729])
    parser.add_argument("--macrel-container", default="amp-macrel")
    parser.add_argument("--hydramp-container", default="amp-hydramp")
    parser.add_argument("--hydramp-model-dir", type=Path, default=root / "data/models/hydramp/37")
    parser.add_argument("--hydramp-script", type=Path, default=root / "services/09-hydramp/scripts/score_amp_classifier_batch.py")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = collect(args.bridge_dir, args.ar_dir, args.seeds)
    frame.to_csv(args.output_dir / "predictor_input_index.csv", index=False)
    fasta = args.output_dir / "predictor_input.fasta"
    write_fasta(frame, fasta)
    macrel = run_macrel(frame, fasta, args.output_dir, args.macrel_container)
    hydramp = run_hydramp(frame, args.output_dir, args.hydramp_container, args.hydramp_script)
    scored = frame.merge(macrel, on="predictor_id", validate="one_to_one").merge(hydramp, on="predictor_id", validate="one_to_one")
    scored["predictor_consensus_available"] = scored["macrel_probability"].notna() & scored["hydramp_probability"].notna()
    scored["predictor_consensus_probability"] = scored[["macrel_probability", "hydramp_probability"]].mean(axis=1, skipna=False)
    scored["predictor_consensus_amp_0_5"] = scored["predictor_consensus_available"] & scored["macrel_probability"].ge(0.5) & scored["hydramp_probability"].ge(0.5)
    scored.to_csv(args.output_dir / "predictor_scores.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    for (implementation, seed), group in scored.groupby(["implementation", "seed"], sort=False):
        consensus = group[group["predictor_consensus_available"]]
        rho = spearmanr(consensus["macrel_probability"], consensus["hydramp_probability"]).statistic if len(consensus) > 1 else float("nan")
        summary_rows.append({
            "implementation": implementation,
            "seed": seed,
            "n": len(group),
            "macrel_coverage": group["macrel_probability"].notna().mean(),
            "macrel_probability_mean": group["macrel_probability"].mean(),
            "macrel_probability_median": group["macrel_probability"].median(),
            "macrel_pass_0_5_rate": group["macrel_probability"].ge(0.5).mean(),
            "hydramp_coverage": group["hydramp_probability"].notna().mean(),
            "hydramp_probability_mean": group["hydramp_probability"].mean(),
            "hydramp_probability_median": group["hydramp_probability"].median(),
            "hydramp_pass_0_5_rate_among_scored": group.loc[group["hydramp_probability"].notna(), "hydramp_probability"].ge(0.5).mean(),
            "two_predictor_coverage": group["predictor_consensus_available"].mean(),
            "two_predictor_amp_consensus_rate_among_scored": consensus["predictor_consensus_amp_0_5"].mean(),
            "macrel_hydramp_spearman": rho,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "predictor_summary.csv", index=False)
    manifest = {
        "status": "complete",
        "input_n": len(frame),
        "macrel_version": subprocess.run(["docker", "exec", args.macrel_container, "macrel", "--version"], capture_output=True, text=True).stdout.strip(),
        "macrel_output_sha256": sha256_file(args.output_dir / "macrel.out.prediction.gz"),
        "hydramp_model_directory_sha256": directory_hash(args.hydramp_model_dir),
        "hydramp_score_sha256": sha256_file(args.output_dir / "hydramp_scores.csv"),
        "predictor_score_sha256": sha256_file(args.output_dir / "predictor_scores.csv"),
        "boundary": "Macrel/HydrAMP agreement is computational predictor consensus, not biological validation.",
    }
    (args.output_dir / "predictor_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
