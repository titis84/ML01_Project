#!/usr/bin/env python3
"""Evaluate ASR transcriptions with WER for the 6 project scenarios.

The script is intentionally dependency-light: it uses only the Python standard
library for WER and matplotlib for figures when available.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path


PIPELINES = ("brut", "denoise", "vad")
VERSIONS = ("original", "corrige")


def normalize_text(text: str) -> list[str]:
    """Normalize English text into word tokens for WER."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return text.split()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance over word tokens."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            if ref_word == hyp_word:
                current.append(previous[j - 1])
            else:
                current.append(
                    1
                    + min(
                        previous[j],      # deletion
                        current[j - 1],   # insertion
                        previous[j - 1],  # substitution
                    )
                )
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> float:
    reference_tokens = normalize_text(reference)
    hypothesis_tokens = normalize_text(hypothesis)
    if not reference_tokens:
        return 0.0 if not hypothesis_tokens else 1.0
    return edit_distance(reference_tokens, hypothesis_tokens) / len(reference_tokens)


def load_librispeech_references(path: Path) -> dict[str, str]:
    references: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        sample_id, text = line.split(maxsplit=1)
        references[sample_id] = text
    return references


def extract_transcribed_text(path: Path) -> str:
    content = path.read_text(encoding="utf-8", errors="replace")
    markers = ("--- TEXTE CORRIG", "--- TEXTE ---")
    for marker in markers:
        index = content.find(marker)
        if index != -1:
            after_marker = content[index:]
            lines = after_marker.splitlines()
            return "\n".join(lines[1:]).strip()
    return content.strip()


def find_transcription_file(base_dir: Path, pipeline: str, sample_id: str, version: str) -> Path | None:
    candidates = sorted((base_dir / pipeline).glob(f"{sample_id}*_{version}.txt"))
    return candidates[0] if candidates else None


def evaluate(references_path: Path, transcriptions_dir: Path, output_dir: Path) -> list[dict[str, str]]:
    references = load_librispeech_references(references_path)
    rows: list[dict[str, str]] = []

    for sample_id, reference_text in sorted(references.items()):
        for pipeline in PIPELINES:
            for version in VERSIONS:
                transcription_path = find_transcription_file(
                    transcriptions_dir, pipeline, sample_id, version
                )
                if transcription_path is None:
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "pipeline": pipeline,
                            "version": version,
                            "wer": "",
                            "reference_words": str(len(normalize_text(reference_text))),
                            "hypothesis_words": "",
                            "transcription_file": "",
                            "status": "missing_transcription",
                        }
                    )
                    continue

                hypothesis = extract_transcribed_text(transcription_path)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "pipeline": pipeline,
                        "version": version,
                        "wer": f"{wer(reference_text, hypothesis):.6f}",
                        "reference_words": str(len(normalize_text(reference_text))),
                        "hypothesis_words": str(len(normalize_text(hypothesis))),
                        "transcription_file": str(transcription_path),
                        "status": "ok",
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "wer_results.csv", rows)
    write_summary(output_dir / "wer_summary.csv", rows)
    write_report(output_dir / "summary.md", rows)
    write_figures(output_dir / "figures", rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "sample_id",
        "pipeline",
        "version",
        "wer",
        "reference_words",
        "hypothesis_words",
        "transcription_file",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scenario_name(row: dict[str, str]) -> str:
    llm = "with_llm" if row["version"] == "corrige" else "without_llm"
    return f"{row['pipeline']}_{llm}"


def grouped_wers(rows: list[dict[str, str]]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row["status"] != "ok" or not row["wer"]:
            continue
        grouped.setdefault(scenario_name(row), []).append(float(row["wer"]))
    return grouped


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    grouped = grouped_wers(rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["scenario", "mean_wer", "median_wer", "sample_count"],
        )
        writer.writeheader()
        for name, values in sorted(grouped.items()):
            writer.writerow(
                {
                    "scenario": name,
                    "mean_wer": f"{statistics.mean(values):.6f}",
                    "median_wer": f"{statistics.median(values):.6f}",
                    "sample_count": len(values),
                }
            )


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    grouped = grouped_wers(rows)
    summary = {
        name: statistics.mean(values)
        for name, values in grouped.items()
        if values
    }
    best = min(summary.items(), key=lambda item: item[1]) if summary else None

    lines = [
        "# WER Evaluation Summary",
        "",
        "This evaluation uses the 15 simple English samples from `121-121726.trans.txt`.",
        "The first overlapped-dialogue batch still needs manually prepared references before WER can be computed fairly.",
        "",
        "## Average WER",
        "",
        "| Scenario | Mean WER |",
        "|---|---:|",
    ]
    for name, value in sorted(summary.items()):
        lines.append(f"| {name} | {value:.3f} |")
    if best:
        lines.extend(
            [
                "",
                f"Best scenario on this dataset: `{best[0]}` with mean WER `{best[1]:.3f}`.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(output_dir: Path, rows: list[dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = grouped_wers(rows)
    if not grouped:
        return

    scenario_order = [
        "brut_without_llm",
        "denoise_without_llm",
        "vad_without_llm",
        "brut_with_llm",
        "denoise_with_llm",
        "vad_with_llm",
    ]
    labels = [name for name in scenario_order if name in grouped]
    means = [statistics.mean(grouped[name]) for name in labels]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, means, color=["#4C78A8", "#72B7B2", "#54A24B", "#F58518", "#E45756", "#B279A2"])
    plt.ylabel("Mean WER")
    plt.title("Average WER across 6 scenarios")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "average_wer_6_scenarios.png", dpi=200)
    plt.close()

    sample_rows = [row for row in rows if row["status"] == "ok" and row["wer"]]
    sample_ids = sorted({row["sample_id"] for row in sample_rows})
    without_llm = {
        pipeline: [
            float(next(row["wer"] for row in sample_rows if row["sample_id"] == sample_id and row["pipeline"] == pipeline and row["version"] == "original"))
            for sample_id in sample_ids
        ]
        for pipeline in PIPELINES
    }
    x_positions = list(range(len(sample_ids)))
    width = 0.25
    plt.figure(figsize=(13, 5))
    for offset, pipeline in zip((-width, 0, width), PIPELINES):
        plt.bar(
            [x + offset for x in x_positions],
            without_llm[pipeline],
            width=width,
            label=pipeline,
        )
    plt.ylabel("WER")
    plt.title("WER by sample without LLM correction")
    plt.xticks(x_positions, sample_ids, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wer_by_sample_without_llm.png", dpi=200)
    plt.close()

    improvements: dict[str, float] = {}
    for pipeline in PIPELINES:
        original_name = f"{pipeline}_without_llm"
        corrected_name = f"{pipeline}_with_llm"
        if original_name in grouped and corrected_name in grouped:
            improvements[pipeline] = statistics.mean(grouped[original_name]) - statistics.mean(grouped[corrected_name])

    plt.figure(figsize=(8, 4))
    plt.axhline(0, color="#333333", linewidth=0.8)
    plt.bar(improvements.keys(), improvements.values(), color="#59A14F")
    plt.ylabel("Mean WER reduction")
    plt.title("Effect of LLM correction")
    plt.tight_layout()
    plt.savefig(output_dir / "llm_correction_improvement.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute WER for ASR project outputs.")
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("evaluation_input/121-121726.trans.txt"),
        help="Path to a LibriSpeech-style .trans.txt reference file.",
    )
    parser.add_argument(
        "--transcriptions",
        type=Path,
        default=Path("evaluation_input/second_try/resultats_transcriptions"),
        help="Path to resultats_transcriptions containing brut/denoise/vad folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/second_try_121-121726"),
        help="Output directory for CSV files and figures.",
    )
    args = parser.parse_args()

    rows = evaluate(args.references, args.transcriptions, args.output)
    ok_count = sum(1 for row in rows if row["status"] == "ok")
    missing_count = len(rows) - ok_count
    print(f"Evaluated {ok_count} transcription files.")
    if missing_count:
        print(f"Missing {missing_count} transcription files.")
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
