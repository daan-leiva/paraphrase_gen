from typing import Tuple

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Creates a tokenizer and sequence to sequence model from the pretrained identifier.
def load_model_and_tokenizer(
    model_name: str,
) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model
