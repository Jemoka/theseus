"""Regenerate boundary plots from a completed benchmark run's output.log.

Parses ``EVAL | {...}`` and ``DATASET | switching primary from X to Y``
lines out of a loguru-formatted log, then drives the same
``_make_eval_bar_chart`` / ``_make_eval_timeline_chart`` helpers used by
the live trainer.  Intended for recovering from runs where
``logging.plots.save`` didn't land plots on disk.

Usage::

    uv run python scripts/recover_benchmark_plots.py \\
        --log ~/Downloads/output.log \\
        --metadata ~/Downloads/metadata.json \\
        --out ./output
"""

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from theseus.experiments.continual.abcd import (
    _make_eval_bar_chart,
    _make_eval_timeline_chart,
)
from theseus.plot import apply_theme


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
EVAL_RE = re.compile(r"EVAL \| (\{[^}]*\})")
DATASET_RE = re.compile(
    r"DATASET \| switching primary from (\d+) to (\d+) at (\d+) tokens"
)
LORA_BOUNDARY_RE = re.compile(r"LORA BOUNDARY \| (\S+) at (\d+) tokens")


def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def parse_log(
    log_path: Path,
) -> Tuple[List[Dict[str, float]], List[Tuple[int, int, int]], List[Tuple[str, int]]]:
    """Pull aggregated EVAL metric dicts and DATASET/LORA boundaries from a log."""
    evals: List[Dict[str, float]] = []
    boundaries: List[Tuple[int, int, int]] = []
    lora_boundaries: List[Tuple[str, int]] = []

    for raw in log_path.read_text().splitlines():
        line = _strip_ansi(raw)

        m = EVAL_RE.search(line)
        if m:
            parsed = ast.literal_eval(m.group(1))
            if isinstance(parsed, dict) and parsed:
                evals.append({str(k): float(v) for k, v in parsed.items()})
            continue

        m = DATASET_RE.search(line)
        if m:
            boundaries.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
            continue

        m = LORA_BOUNDARY_RE.search(line)
        if m:
            lora_boundaries.append((m.group(1), int(m.group(2))))

    return evals, boundaries, lora_boundaries


def save_figures(figures: Dict[str, Any], out_dir: Path, step: int) -> List[Path]:
    written: List[Path] = []
    safe_re = re.compile(r"[^\w\-.]")
    for name, fig in figures.items():
        safe_name = safe_re.sub("_", name)
        path = out_dir / f"{safe_name}_step{step}.pdf"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="output.log path")
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="metadata.json path (for batch/block size → step conversion)",
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()

    meta = json.loads(args.metadata.read_text())
    batch_size = int(meta["config"]["training"]["batch_size"])
    block_size = int(meta["config"]["architecture"]["block_size"])
    tokens_per_step = batch_size * block_size

    evals, boundaries, lora_boundaries = parse_log(args.log)

    if not evals:
        raise SystemExit(f"no EVAL | {{...}} lines found in {args.log}")

    args.out.mkdir(parents=True, exist_ok=True)

    # Configure matplotlib with the same theme the live trainer uses so
    # recovered PDFs match runtime-saved ones.
    import matplotlib

    matplotlib.use("Agg")
    apply_theme()

    # Pair each EVAL with the boundary that immediately follows it.  On
    # shutdown-truncated logs the lists can end with an extra EVAL; we
    # stop at whichever is shorter.
    n = min(len(evals), max(len(boundaries), len(lora_boundaries)))
    if n == 0:
        raise SystemExit(
            "found EVAL lines but no DATASET/LORA BOUNDARY transitions to label them"
        )

    eval_history: Dict[str, List[Tuple[int, float]]] = {}
    boundary_tokens: List[int] = []
    written: List[Path] = []

    use_lora = len(lora_boundaries) >= len(boundaries)
    timeline_key = "eval/lora_timeline" if use_lora else "eval/timeline"

    for i in range(n):
        metrics = evals[i]

        if use_lora:
            label, ntok = lora_boundaries[i]
            boundary_label = f"lora_{label}"
        else:
            old_idx, new_idx, ntok = boundaries[i]
            boundary_label = f"{old_idx}_to_{new_idx}"

        step = ntok // tokens_per_step

        # Per-boundary bar charts (already split by metric group).
        bar_figs = _make_eval_bar_chart(metrics, boundary_label)
        written.extend(save_figures(bar_figs, args.out, step))

        # Accumulate timeline history and emit the rolling timeline snapshot.
        boundary_tokens.append(ntok)
        for k, v in metrics.items():
            eval_history.setdefault(k, []).append((ntok, v))

        timeline_figs = _make_eval_timeline_chart(
            {k: list(v) for k, v in eval_history.items()},
            list(boundary_tokens),
            timeline_key=timeline_key,
        )
        written.extend(save_figures(timeline_figs, args.out, step))

    print(f"wrote {len(written)} figures to {args.out}")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
