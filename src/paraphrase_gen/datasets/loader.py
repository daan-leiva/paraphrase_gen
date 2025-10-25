from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


# Load paraphrase wrapper used by other classes to do a load_dataset function without import datasets
# takes the dataset repository name and the dataset configuation name as inputs
# returns a DatasetDict
def load_paraphrase_dataset(name: str, config: str) -> DatasetDict:
    return load_dataset(name, config)


# Helper to identify paraphrase samples
def _is_positive(example, label_col: str) -> bool:
    # QQP positives are labeled 1
    return example[label_col] == 1


def prepare_t5_format(
    ds: DatasetDict,
    text_col: str,
    pair_col: str,
    label_col: str,
    bidirectional: bool = False,
) -> DatasetDict:
    """
    Converts a paired classification dataset into text-to-text format:
      input_text: "paraphrase: {text}"
      target_text: "{pair}"
    Keeps only positive (paraphrase) pairs, and skips splits without labels (e.g., test).

    Parameters
    ----------
    ds : DatasetDict
        raw DatasetDict containing all the splits
    text_col : str
        Name of the first sentence
    pair_col : str
        Name of the second sentence
    label_col : str
        Label column used to identify paraphrased pairs (== 1)
    bidirectional : bool
        Bit used to increase training data by reversing the sentence pair
    """
    output_dataset_dict = DatasetDict()

    for split, dset in ds.items():
        # Skip splits without labels
        if label_col not in dset.column_names:
            continue

        # Keep only positive pairs using Dataset.filter
        positive_dataset: Dataset = dset.filter(
            lambda example: _is_positive(example, label_col), batched=False
        )

        # Map to text-to-text fields
        def to_t5(batch: Dict):
            inputs = [f"paraphrase: {a}" for a in batch[text_col]]
            targets = list(batch[pair_col])

            if bidirectional:
                # Add the reverse direction to increase data
                inputs += [f"paraphrase: {b}" for b in batch[pair_col]]
                targets += list(batch[text_col])

            return {"input_text": inputs, "target_text": targets}

        # apply the t5 map function so there are only input and target text columns
        mapped = positive_dataset.map(
            to_t5,
            batched=True,
            remove_columns=positive_dataset.column_names,  # drop original dataset columns
        )
        output_dataset_dict[split] = mapped

    # error if nothing is produced
    if not output_dataset_dict:
        raise ValueError(
            "prepare_t5_format produced no splits. "
            "Check that the dataset contains the label column and the adapter arguments are correct."
        )

    return output_dataset_dict


# contains [HuggingFace repo name, dataset config, text column name, paired column name, label column name]
MixtureItem = Tuple[str, str, str, str, str]


# similar as above but combines all of the datasets
# deals with shuffling size caps and missing splits
def build_paraphrase_mixture(
    items: List[MixtureItem],
    bidirectional: bool = True,
    train_take: int = 300_000,
    val_take: int = 6_000,
) -> DatasetDict:
    trains, vals = [], []
    for repo_name, config, text_col, pair_col, label_col in items:
        raw = load_paraphrase_dataset(repo_name, config)
        processed_dataset = prepare_t5_format(
            raw,
            text_col=text_col,
            pair_col=pair_col,
            label_col=label_col,
            bidirectional=bidirectional,
        )

        # check for a train dataset
        if processed_dataset["train"] is None:
            raise KeyError(
                f"{repo_name}/{config}: missing required split 'train'. "
                f"Available: {list(processed_dataset.keys())}"
            )
        trains.append(processed_dataset["train"])

        # attempt to extract validation dataset
        v = (
            processed_dataset.get("validation")
            or processed_dataset.get("validation_matched")
            or processed_dataset.get("validation_mismatched")
        )
        # if no validation dataset split the train dataset
        if v is None:
            split = processed_dataset["train"].train_test_split(test_size=0.02, seed=13)
            v = split["test"]
            trains[-1] = split[
                "train"
            ]  # replace the last added train dataset with our new train dataset
        vals.append(v)

    # concatenate
    train_mix = concatenate_datasets(trains)
    val_mix = concatenate_datasets(vals)

    # shuffle before capping to avoid positional bias
    train_mix = train_mix.shuffle(seed=13)
    val_mix = val_mix.shuffle(seed=13)

    # limit data by the take limiters vars
    if train_take and len(train_mix) > train_take:
        train_mix = train_mix.select(range(train_take))
    if val_take and len(val_mix) > val_take:
        val_mix = val_mix.select(range(val_take))

    return DatasetDict(train=train_mix, validation=val_mix)
