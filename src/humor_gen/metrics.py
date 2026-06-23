from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from humor_gen.utils import read_jsonl


def load_generated_dir(path: str) -> list[dict[str, Any]]:
    # rglob so nested layouts like data/generated/{baseline,rag}/*.jsonl are picked up.
    rows: list[dict[str, Any]] = []
    for file in sorted(Path(path).rglob("*.jsonl")):
        rows.extend(read_jsonl(file))
    return rows


def load_judged_dir(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file in sorted(Path(path).rglob("*.jsonl")):
        rows.extend(read_jsonl(file))
    return rows


def generation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("model", ""), row.get("method", ""))].append(row)
    summary = []
    for (model, method), group in sorted(grouped.items()):
        total = len(group)
        valid = sum(bool(row.get("valid")) for row in group)
        lengths = [len(str(row.get("generated_joke", "")).split()) for row in group]
        summary.append(
            {
                "model": model,
                "method": method,
                "total": total,
                "valid": valid,
                "invalid": total - valid,
                "constraint_satisfaction_rate": valid / total if total else 0.0,
                "invalid_output_rate": (total - valid) / total if total else 0.0,
                "average_joke_length": sum(lengths) / total if total else 0.0,
            }
        )
    return summary


def win_rate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    appearances = Counter()
    wins = Counter()
    ties = Counter()
    for row in rows:
        for model in (row.get("model_a"), row.get("model_b")):
            if model:
                appearances[model] += 1
        winner = row.get("winner")
        if winner == "tie":
            for model in (row.get("model_a"), row.get("model_b")):
                if model:
                    ties[model] += 1
        elif winner:
            wins[winner] += 1
    return [
        {
            "model": model,
            "appearances": appearances[model],
            "wins": wins[model],
            "ties": ties[model],
            "win_rate": wins[model] / appearances[model] if appearances[model] else 0.0,
            "tie_rate": ties[model] / appearances[model] if appearances[model] else 0.0,
        }
        for model in sorted(appearances)
    ]


def write_csv(rows: list[dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def create_figures(generation_rows: list[dict[str, Any]], judged_rows: list[dict[str, Any]], output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gen_summary = generation_summary(generation_rows)
    win_summary = win_rate_summary(judged_rows)
    _bar(
        [row["model"] for row in win_summary],
        [row["win_rate"] for row in win_summary],
        "Win rate per model",
        output / "win_rate_per_model.png",
        ylabel="Win rate",
    )
    _bar(
        [f"{row['model']}-{row['method']}" for row in gen_summary],
        [row["constraint_satisfaction_rate"] for row in gen_summary],
        "Constraint satisfaction",
        output / "constraint_satisfaction.png",
        ylabel="Rate",
    )
    method_invalid = defaultdict(list)
    for row in gen_summary:
        method_invalid[row["method"]].append(row["invalid_output_rate"])
    _bar(
        sorted(method_invalid),
        [sum(method_invalid[m]) / len(method_invalid[m]) for m in sorted(method_invalid)],
        "Invalid output rate by method",
        output / "invalid_rate_by_method.png",
        ylabel="Rate",
    )


def _bar(labels: list[str], values: list[float], title: str, path: Path, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _bar_svg(labels, values, title, path.with_suffix(".svg"), ylabel)
        return
    plt.figure(figsize=(8, 4.5))
    if labels:
        plt.bar(labels, values, color=["#356d80", "#d18f3f", "#6a994e", "#8a5a83"][: len(labels)])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0, 1 if values and max(values) <= 1 else None)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _bar_svg(labels: list[str], values: list[float], title: str, path: Path, ylabel: str) -> None:
    width, height = 820, 460
    margin = 70
    max_value = max(values) if values else 1
    max_value = max(max_value, 1)
    bar_width = 42 if labels else 0
    gap = 26
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial" font-size="22">{_escape(title)}</text>',
        f'<text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" text-anchor="middle" font-family="Arial" font-size="13">{_escape(ylabel)}</text>',
    ]
    x = margin
    chart_height = height - 2 * margin
    for label, value in zip(labels, values):
        bar_height = int((value / max_value) * chart_height)
        y = height - margin - bar_height
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="#356d80"/>')
        parts.append(f'<text x="{x + bar_width / 2}" y="{y - 8}" text-anchor="middle" font-family="Arial" font-size="12">{value:.2f}</text>')
        parts.append(f'<text x="{x + bar_width / 2}" y="{height - 38}" text-anchor="end" transform="rotate(-35 {x + bar_width / 2} {height - 38})" font-family="Arial" font-size="11">{_escape(label)}</text>')
        x += bar_width + gap
    parts.append(f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
