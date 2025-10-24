import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field, confloat, conint

from paraphrase_gen.inference.generate import Paraphraser

# Environment defaults for quick startup
MODEL_PATH = os.getenv("CHECKPOINT_DIR", "runs/t5_small_qqp")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "64"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "4"))

# Initialize the generator once at process start
paraphraser = Paraphraser(
    MODEL_PATH,
    max_new_tokens=MAX_NEW_TOKENS,
    num_beams=NUM_BEAMS,
)


class ParaphraseRequest(BaseModel):
    inputs: List[str] = Field(..., description="List of strings to paraphrase")
    # Optional decoding controls for per request overrides
    n_best: Optional[conint(ge=1)] = None
    max_new_tokens: Optional[conint(ge=1)] = None
    min_new_tokens: Optional[conint(ge=0)] = None
    num_beams: Optional[conint(ge=1)] = None
    do_sample: Optional[bool] = None
    temperature: Optional[confloat(gt=0.0)] = None
    top_p: Optional[confloat(gt=0.0, le=1.0)] = None
    top_k: Optional[conint(ge=0)] = None
    no_repeat_ngram_size: Optional[conint(ge=0)] = None
    repetition_penalty: Optional[confloat(gt=0.0)] = None
    early_stopping: Optional[bool] = None


class ParaphraseResponse(BaseModel):
    outputs: List[str]


app = FastAPI(title="Paraphrase API")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/paraphrase", response_model=ParaphraseResponse)
def paraphrase(req: ParaphraseRequest) -> ParaphraseResponse:
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
    outputs = paraphraser.generate(req.inputs, overrides=overrides)
    return ParaphraseResponse(outputs=outputs)
