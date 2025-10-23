import argparse

import numpy as np
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from paraphrase_gen.data.datasets import load_paraphrase_dataset, prepare_t5_format
from paraphrase_gen.models.modeling import load_model_and_tokenizer
from paraphrase_gen.training.metrics import compute_metrics


def tokenize_function(batch, tokenizer, max_length):
    """
    Tokenizes input and target text for a text to text model
    """
    model_inputs = tokenizer(
        batch["input_text"], max_length=max_length, truncation=True
    )
    labels = tokenizer(
        text_target=batch["target_text"], max_length=max_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="t5-small")
    parser.add_argument("--dataset_name", default="glue")
    parser.add_argument("--dataset_config", default="qqp")
    parser.add_argument("--output_dir", default="runs/t5_small_qqp")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    ds_raw = load_paraphrase_dataset(args.dataset_name, args.dataset_config)
    ds_proc = prepare_t5_format(
        ds_raw, text_col="question1", pair_col="question2", label_col="label"
    )

    tokenizer, model = load_model_and_tokenizer(args.model_name)

    tokenized = ds_proc.map(
        lambda b: tokenize_function(b, tokenizer, args.max_length),
        batched=True,
        remove_columns=ds_proc["train"].column_names,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        logging_steps=50,
        report_to=["none"],
    )

    eval_ds = (
        tokenized.get("validation")
        or tokenized.get("validation_matched")
        or tokenized.get("validation_mismatched")
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=lambda p: compute_metrics(
            (
                tokenizer.batch_decode(p.predictions, skip_special_tokens=True),
                tokenizer.batch_decode(
                    np.where(p.label_ids != -100, p.label_ids, tokenizer.pad_token_id),
                    skip_special_tokens=True,
                ),
            )
        ),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
