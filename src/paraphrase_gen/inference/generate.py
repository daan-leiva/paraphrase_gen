from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Paraphraser:
    """
    Loads a sequence to sequence checkpoint and generates paraphrases.
    Decoding parameters are configurable at construction and per call.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 64,
        min_new_tokens: int = 0,
        num_beams: int = 4,
        n_best: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
        early_stopping: bool = True,
    ) -> None:
        # Lazy device selection for CPU or CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(
            self.device
        )

        # Store default decoding parameters
        self.defaults: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "num_beams": num_beams,
            "num_return_sequences": max(1, n_best),
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repetition_penalty": repetition_penalty,
            "early_stopping": early_stopping,
        }

    def generate(
        self, texts: List[str], overrides: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generates paraphrases for a list of input strings.
        Parameters can be overridden per call through the overrides map.
        """
        prompts = [f"paraphrase: {t}" for t in texts]
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        params = dict(self.defaults)
        if overrides:
            params.update({k: v for k, v in overrides.items() if v is not None})

        # Ensure num_return_sequences vs beams is valid when not sampling
        if not params.get("do_sample", False):
            nret = int(params.get("num_return_sequences", 1))
            nbeams = int(params.get("num_beams", 1))
            if nret > nbeams:
                params["num_return_sequences"] = nbeams
            # Drop sampling-only knobs to avoid warnings
            for k in ("temperature", "top_p", "top_k"):
                params.pop(k, None)

        out_ids = self.model.generate(**enc, **params)
        outputs = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        return outputs
