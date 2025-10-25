import os
from functools import lru_cache
from typing import Annotated, Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from paraphrase_gen.inference.generate import Paraphraser

# Base configuration from environment with safe defaults
MODEL_PATH = os.getenv("CHECKPOINT_DIR", "runs/t5_small_mix")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "64"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "4"))

print(f"Default model path: {MODEL_PATH}")


# Reusable loader with simple in-process cache
@lru_cache(maxsize=8)
def get_paraphraser(
    model_path: str, max_new_tokens: int, num_beams: int
) -> Paraphraser:
    return Paraphraser(model_path, max_new_tokens=max_new_tokens, num_beams=num_beams)


# Schemas
class ParaphraseRequest(BaseModel):
    inputs: List[str] = Field(..., description="List of strings to paraphrase")
    model_path: str | None = Field(
        default=None,
        description="Optional checkpoint dir. Defaults to CHECKPOINT_DIR.",
    )

    n_best: Annotated[int, Field(ge=1)] | None = None
    max_new_tokens: Annotated[int, Field(ge=1)] | None = None
    min_new_tokens: Annotated[int, Field(ge=0)] | None = None
    num_beams: Annotated[int, Field(ge=1)] | None = None
    do_sample: bool | None = None
    temperature: Annotated[float, Field(gt=0.0)] | None = None
    top_p: Annotated[float, Field(gt=0.0, le=1.0)] | None = None
    top_k: Annotated[int, Field(ge=0)] | None = None
    no_repeat_ngram_size: Annotated[int, Field(ge=0)] | None = None
    repetition_penalty: Annotated[float, Field(gt=0.0)] | None = None
    early_stopping: bool | None = None


class ParaphraseResponse(BaseModel):
    outputs: List[str]


# App
app = FastAPI(title="Paraphrase API")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/paraphrase", response_model=ParaphraseResponse)
def paraphrase(req: ParaphraseRequest) -> ParaphraseResponse:
    # Resolve which model to use for this request
    model_path = req.model_path or MODEL_PATH
    para = get_paraphraser(model_path, MAX_NEW_TOKENS, NUM_BEAMS)

    # Map user supplied decoding controls to model.generate kwargs
    overrides: Dict[str, Any] = {
        "num_return_sequences": req.n_best,
        "max_new_tokens": req.max_new_tokens,
        "min_new_tokens": req.min_new_tokens,
        "num_beams": req.num_beams,
        "do_sample": req.do_sample,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "no_repeat_ngram_size": req.no_repeat_ngram_size,
        "repetition_penalty": req.repetition_penalty,
        "early_stopping": req.early_stopping,
    }

    outputs = para.generate(req.inputs, overrides=overrides)
    return ParaphraseResponse(outputs=outputs)
