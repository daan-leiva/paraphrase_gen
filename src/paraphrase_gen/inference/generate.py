from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Paraphraser:
    """
    Load a sequence-to-sequence checkpoint and generates paraphrases.
    Decoding parameters are configurable for stability or diversity.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 64,
        num_beams: int = 4,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.temperature = temperature

    def generate(self, texts: List[str]) -> List[str]:
        prompts = [f"paraphrase: {t}" for t in texts]
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            do_sample=self.do_sample,
            temperature=self.temperature,
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)
