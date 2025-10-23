from datasets import DatasetDict, load_dataset


def load_paraphrase_dataset(name: str, config: str) -> DatasetDict:
    """
    Load a dataset from Hugging Face Datasets by repository name and configuration.

    Parameters
    ----------
    name : str
        Dataset repository name ("glue").
    config : str
        Dataset configuration ("qqp").

    Returns
    -------
    DatasetDict
        A mapping of split name to dataset, including "train",
        "validation", and possibly "test".
    """
    return load_dataset(name, config)


def prepare_t5_format(
    ds: DatasetDict, text_col: str, pair_col: str, label_col: str
) -> DatasetDict:
    """
    Convert a pairwise classification dataset into text to text fields for T5 style models.

    The function keeps only positive paraphrase pairs and produces two string features:
    "input_text" containing the task prefix and source text, and "target_text"
    containing the paired paraphrase target.

    Parameters
    ----------
    ds : DatasetDict
        Source dataset with standard splits.
    text_col : str
        Column name for the first text in the pair.
    pair_col : str
        Column name for the second text in the pair.
    label_col : str
        Column name for the binary label where 1 denotes a paraphrase pair
        such as in GLUE QQP.

    Returns
    -------
    DatasetDict
        A new DatasetDict with the same splits as the input. Each split contains
        only "input_text" and "target_text" features after mapping.

    Notes
    -----
    - Input rows with label equal to 1 are kept. All others are dropped.
    - The task prefix "paraphrase:" is prepended to the source text to follow
      common T5 prompting practice.
    - All original columns are removed in the mapped result to keep only the
      model ready fields.
    """

    def to_t5(batch):
        # Build per batch lists for new features expected by text to text models
        inp = []
        tgt = []
        # Zip over the three required columns in the current batch
        for a, b, y in zip(batch[text_col], batch[pair_col], batch[label_col]):
            # Keep only paraphrase pairs (label_col == 1)
            if y == 1:
                # Prefix the source with a T5 style task hint
                inp.append(f"paraphrase: {a}")
                # Use the paired text as the training target
                tgt.append(b)
        # Return a dict of new columns (Datasets.map will merge)
        return {"input_text": inp, "target_text": tgt}

    out = DatasetDict()
    for split in ds:
        # Apply batch mapping. Only input_text and target_text
        # remain in the transformed split.
        out[split] = ds[split].map(
            to_t5, batched=True, remove_columns=ds[split].column_names
        )

    return out
