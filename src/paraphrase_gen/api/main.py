import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from paraphrase_gen.inference.generate import Paraphraser

MODEL_PATH = os.getenv("CHECKPOINT_DIR", "runs/t5_small_qqp")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "64"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "4"))

paraphraser = Paraphraser(
    MODEL_PATH, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS
)
app = FastAPI(title="Paraphrase API")


class ParaphraseRequest(BaseModel):
    inputs: List[str]


class ParaphraseResponse(BaseModel):
    outputs: List[str]


@app.get("/health")
@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.post("/paraphrase", response_model=ParaphraseResponse)
def paraphrase(req: ParaphraseRequest):
    outputs = paraphraser.generate(req.inputs)
    return ParaphraseResponse(outputs=outputs)
