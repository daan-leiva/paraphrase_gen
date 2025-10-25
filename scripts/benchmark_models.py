import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

MODELS = [
    ("t5_small_mix", "runs/t5_small_mix"),
    ("t5_small_qqp", "runs/t5_small_qqp"),
    ("t5_small_mrpc", "runs/t5_small_mrpc"),
    ("t5_small_paws", "runs/t5_small_paws"),
]

OUT_BASE = Path("artifacts/bench")
NUM_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 1000


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, path in MODELS:
        out_dir = OUT_BASE / name
        cmd = [
            sys.executable,
            "-m",
            "paraphrase_gen.evaluation.evaluate",
            "--model_path",
            path,
            "--num_samples",
            str(NUM_SAMPLES),
            "--out_dir",
            str(out_dir),
            "--max_new_tokens",
            "48",
            "--num_beams",
            "4",
            "--n_best",
            "1",
        ]
        subprocess.run(cmd, check=True)
        with open(out_dir / "metrics.json", "r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append(
            {
                "model": name,
                "count": m["count"],
                "bleu": m["bleu"],
                "rouge1": m["rouge1"],
                "rougeL": m["rougeL"],
            }
        )
    df = pd.DataFrame(rows).sort_values(["bleu", "rougeL"], ascending=False)
    csv_path = OUT_BASE / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(df)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
