import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from paraphrase_gen.datasets.loader import (
    build_paraphrase_mixture,
    load_paraphrase_dataset,
    prepare_t5_format,
)
from paraphrase_gen.models.modeling import load_model_and_tokenizer
from paraphrase_gen.training.metrics import compute_metrics as _compute


# Tokenizes input and target text for a text to text model
def tokenize_function(batch, tokenizer, max_length):
    model_inputs = tokenizer(
        batch["input_text"], max_length=max_length, truncation=True
    )
    labels = tokenizer(
        text_target=batch["target_text"], max_length=max_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Generates a paraphrase of a fixed text after each epoch and writes it to a file.
# This is intended for quick qualitative inspection during training.
class SpotCheckCallback(TrainerCallback):
    def __init__(self, tokenizer, text: str, out_path: Path, gen_kwargs: dict):
        self.tokenizer = tokenizer
        self.text = text
        self.out_path = out_path
        self.gen_kwargs = gen_kwargs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure parent directory exists
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()
        prompt = f"paraphrase: {self.text}"
        enc = self.tokenizer(
            [prompt], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        model.to(self.device)
        with torch.no_grad():
            out_ids = model.generate(**enc, **self.gen_kwargs)
        out_text = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]

        line = f"epoch={int(state.epoch)}\tinput={self.text}\toutput={out_text}\n"
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(line)

        # Also print to console
        print(f"[spotcheck] {line.strip()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="t5-small")
    parser.add_argument("--dataset_name", default="glue")
    parser.add_argument("--dataset_config", default="qqp")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--text_col", default=None)
    parser.add_argument("--pair_col", default=None)
    parser.add_argument("--label_col", default="label")
    parser.add_argument(
        "--spot_text",
        default="I will follow up later today once I review the latest results from the test run.",
    )
    parser.add_argument("--spot_num_beams", type=int, default=4)
    parser.add_argument("--spot_max_new_tokens", type=int, default=48)
    parser.add_argument("--spot_min_new_tokens", type=int, default=6)
    parser.add_argument("--spot_no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--spot_repetition_penalty", type=float, default=1.05)
    parser.add_argument("--spot_do_sample", action="store_true")
    parser.add_argument("--spot_top_p", type=float, default=0.9)
    parser.add_argument("--spot_temperature", type=float, default=0.8)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint in output_dir if present.",
    )
    parser.add_argument(
        "--mix_default",
        action="store_true",
        help="Use a default QQP+MRPC+PAWS mixture instead of a single dataset",
    )
    parser.add_argument("--train_take", type=int, default=300_000)
    parser.add_argument("--val_take", type=int, default=6_000)

    args = parser.parse_args()

    # fix default path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = Path("runs") / f"t5_small_{args.dataset_config}_{timestamp}"
    output_dir = args.output_dir or default_output

    if args.mix_default:
        ds_proc = build_paraphrase_mixture(
            items=[
                ("glue", "qqp", "question1", "question2", "label"),
                ("glue", "mrpc", "sentence1", "sentence2", "label"),
                ("paws", "labeled_final", "sentence1", "sentence2", "label"),
            ],
            bidirectional=True,
            train_take=args.train_take,
            val_take=args.val_take,
        )
    else:
        ds_raw = load_paraphrase_dataset(args.dataset_name, args.dataset_config)

        cols = ds_raw["train"].column_names
        text_col = args.text_col
        pair_col = args.pair_col
        label_col = args.label_col

        # try to infer text and pair column name
        if text_col is None or pair_col is None:
            if {"question1", "question2"}.issubset(cols):
                text_col, pair_col = "question1", "question2"  # QQP
            elif {"sentence1", "sentence2"}.issubset(cols):
                text_col, pair_col = "sentence1", "sentence2"  # MRPC, PAWS
            else:
                raise ValueError(
                    f"Could not infer text columns from {cols}. Pass --text_col and --pair_col explicitly."
                )

        # transform the data into t5 format pairs
        ds_proc = prepare_t5_format(
            ds_raw, text_col=text_col, pair_col=pair_col, label_col=label_col
        )

    tokenizer, model = load_model_and_tokenizer(args.model_name)

    tokenized = DatasetDict()
    for split, dset in ds_proc.items():
        tokenized[split] = dset.map(
            lambda b: tokenize_function(b, tokenizer, args.max_length),
            batched=True,
            remove_columns=dset.column_names,  # remove only this split’s columns
        )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Spot check configuration
    spot_kwargs = {
        "max_new_tokens": args.spot_max_new_tokens,
        "min_new_tokens": args.spot_min_new_tokens,
        "num_beams": args.spot_num_beams,
        "no_repeat_ngram_size": args.spot_no_repeat_ngram_size,
        "repetition_penalty": args.spot_repetition_penalty,
    }
    # Remove sampling-only keys unless explicitly requested
    if args.spot_do_sample:
        spot_kwargs.update(
            {
                "do_sample": True,
                "top_p": args.spot_top_p,
                "temperature": args.spot_temperature,
                "num_return_sequences": 1,
            }
        )

    spot_path = Path(output_dir) / "spotcheck.txt"
    spot_cb = SpotCheckCallback(
        tokenizer=tokenizer,
        text=args.spot_text,
        out_path=spot_path,
        gen_kwargs=spot_kwargs,
    )

    train_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=args.max_length,
        logging_steps=50,
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        warmup_ratio=0.1,  # for overfitting issue
        lr_scheduler_type="cosine",  # for overfitting issue
    )

    eval_ds = (
        tokenized.get("validation")
        or tokenized.get("validation_matched")
        or tokenized.get("validation_mismatched")
    )

    if eval_ds is None:
        print("No validation split found")

    # converts model outputs and labels into text and then calls metric function
    def make_compute(tokenizer):
        pad_id = tokenizer.pad_token_id

        def _to_int_token_ids(arr):
            x = np.asarray(arr)

            # If logits in [B, T, V] shape take argmax to get ids
            if x.ndim == 3:
                x = x.argmax(axis=-1)

            # If not integer dtype, cast to int64
            if not np.issubdtype(x.dtype, np.integer):
                # guard against NaNs or weird floats
                x = np.nan_to_num(x, nan=-100.0, posinf=-100.0, neginf=-100.0)
                x = np.round(x).astype(np.int64, copy=False)

            # Replace negatives with pad_id for tokenizer
            if (x < 0).any():
                x = np.where(x < 0, pad_id, x)

            return x

        def compute_for_trainer(eval_pred):
            preds, labels = eval_pred

            # unwrap for outer dimension
            if isinstance(preds, (tuple, list)):
                preds = preds[0]

            preds = _to_int_token_ids(preds)
            pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)

            labels = np.asarray(labels)
            # put back pad where ignore index is used
            labels = np.where(labels != -100, labels, pad_id)
            labels = labels.astype(np.int64, copy=False)
            label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

            return _compute((pred_texts, label_texts))

        return compute_for_trainer

    compute_fn = make_compute(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_fn,
        callbacks=[spot_cb, EarlyStoppingCallback(early_stopping_patience=3)],
    )

    resume_flag = False
    if args.resume:
        # resume if there is any checkpoint under output_dir
        ckpts = [p for p in Path(output_dir).glob("checkpoint-*")]
        resume_flag = len(ckpts) > 0

    trainer.train(resume_from_checkpoint=resume_flag)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
