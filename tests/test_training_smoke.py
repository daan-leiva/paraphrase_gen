import pytest
from transformers import AutoTokenizer

from paraphrase_gen.datasets.loader import load_paraphrase_dataset, prepare_t5_format


@pytest.mark.network
def test_dataset_adapter_has_records():
    ds = load_paraphrase_dataset("glue", "qqp")
    proc = prepare_t5_format(ds, "question1", "question2", "label")
    assert len(proc["train"]) > 0
    rec = proc["train"][0]
    assert "input_text" in rec and "target_text" in rec
    assert isinstance(rec["input_text"], str) and isinstance(rec["target_text"], str)


def test_tokenizer_round_trip():
    tok = AutoTokenizer.from_pretrained("t5-small")
    sample = {"input_text": ["paraphrase: hello"], "target_text": ["hi"]}
    enc = tok(sample["input_text"], truncation=True)
    lab = tok(text_target=sample["target_text"], truncation=True)
    assert "input_ids" in enc and "input_ids" in lab
