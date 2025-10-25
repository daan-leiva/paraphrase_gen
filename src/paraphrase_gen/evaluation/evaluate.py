import argparse
import json
from pathlib import Path
from typing import List

import evaluate
import pandas as pd
from datasets import Dataset, DatasetDict

from paraphrase_gen.datasets.loader import load_paraphrase_dataset, prepare_t5_format
from paraphrase_gen.inference.generate import Paraphraser


# Tries to select a validation like split when available
# otherwise create a small holdout from train.
def pick_validation_split(
    ds: DatasetDict, seed: int = 13, frac: float = 0.02
) -> Dataset:
    for name in ("validation", "validation_matched", "validation_mismatched"):
        if name in ds:
            return ds[name]
    # if no train throw an error
    if "train" not in ds:
        raise KeyError("No 'validation' and no 'train' split available.")
    print(
        f"[eval] No validation split found; creating a {int(frac * 100)}% holdout from train."
    )
    # split from train if not valid split was found
    split = ds["train"].train_test_split(test_size=frac, seed=seed)
    return split["test"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_name", default="glue")
    parser.add_argument("--dataset_config", default="qqp")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--out_dir", default="artifacts/eval_run")
    # Decoding controls
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--min_new_tokens", type=int, default=6)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--n_best", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--early_stopping", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data and adapt format
    ds_raw = load_paraphrase_dataset(args.dataset_name, args.dataset_config)

    # infer column names for datasets (QQP/MRPC/PAWS)
    cols = set(ds_raw["train"].column_names)
    if {"question1", "question2", "label"}.issubset(cols):
        text_col, pair_col, label_col = "question1", "question2", "label"
    elif {"sentence1", "sentence2", "label"}.issubset(cols):
        text_col, pair_col, label_col = "sentence1", "sentence2", "label"
    else:
        raise ValueError(
            f"Could not infer text/label columns from {sorted(cols)}; "
            "expected QQP or MRPC/PAWS style columns."
        )
    print(
        f"[eval] Using columns: text_col={text_col}, pair_col={pair_col}, label_col={label_col}"
    )

    ds_proc = prepare_t5_format(
        ds_raw, text_col=text_col, pair_col=pair_col, label_col=label_col
    )

    val = pick_validation_split(ds_proc)
    # shuffle data so that the multiple datasets (if multiple are used) get recombined
    val = val.shuffle(seed=13)

    n = min(args.num_samples, len(val))
    inputs: List[str] = [
        val[i]["input_text"].replace("paraphrase: ", "", 1) for i in range(n)
    ]
    refs: List[str] = [val[i]["target_text"] for i in range(n)]

    # Initialize generator
    paraphraser = Paraphraser(
        args.model_path,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        num_beams=args.num_beams,
        n_best=args.n_best,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        early_stopping=args.early_stopping,
    )

    # Generate predictions
    preds = paraphraser.generate(inputs)

    # Compute metrics
    bleu = evaluate.load("bleu").compute(
        predictions=preds,
        references=[[r] for r in refs],
    )

    rouge = evaluate.load("rouge").compute(predictions=preds, references=refs)

    metrics = {
        "count": n,
        "bleu": float(bleu.get("bleu", 0.0)),
        "rouge1": float(rouge.get("rouge1", 0.0)),
        "rougeL": float(rouge.get("rougeL", 0.0)),
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "text_col": text_col,
            "pair_col": pair_col,
            "label_col": label_col,
        },
        "settings": {
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "num_beams": args.num_beams,
            "n_best": args.n_best,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "repetition_penalty": args.repetition_penalty,
            "early_stopping": args.early_stopping,
        },
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save a small table of examples
    df = pd.DataFrame({"input": inputs, "reference": refs, "prediction": preds})
    df.to_csv(out_dir / "samples.csv", index=False)

    print(f"Saved metrics to {out_dir / 'metrics.json'}")
    print(f"Saved samples to {out_dir / 'samples.csv'}")


if __name__ == "__main__":
    main()
