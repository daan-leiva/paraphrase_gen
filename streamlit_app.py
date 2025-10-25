# streamlit_app.py
import os
from pathlib import Path

import streamlit as st
import torch

from paraphrase_gen.inference.generate import Paraphraser

# Configure Streamlit before other calls
st.set_page_config(page_title="Paraphrase Demo", layout="centered")

# Setup for running on Render hosting
torch.set_num_threads(max(1, os.cpu_count() // 2))

# to decide model hosting points
RUNNING_ON_RENDER = os.getenv("RENDER", "false").lower() == "true"


# Cache the Paraphraser
@st.cache_resource(show_spinner=False)
def load_paraphraser(model_path: str):
    max_new = int(os.getenv("MAX_NEW_TOKENS", "64"))
    num_beams = int(os.getenv("NUM_BEAMS", "4"))
    return Paraphraser(model_path, max_new_tokens=max_new, num_beams=num_beams)


# Discover available run dirs and build a labeled selector
def available_models():
    base = Path("runs")
    candidates = [
        ("t5_small_mix", base / "t5_small_mix"),
        ("t5_small_qqp", base / "t5_small_qqp"),
        ("t5_small_mrpc", base / "t5_small_mrpc"),
        ("t5_small_paws", base / "t5_small_paws"),
    ]
    # keep only those that exist and look like model dirs
    out = []
    for label, p in candidates:
        if p.is_dir() and (
            (p / "config.json").exists() or (p / "model.safetensors").exists()
        ):
            out.append((label, str(p)))
    return out


if RUNNING_ON_RENDER:
    # Use Hugging Face models if local checkpoints aren’t available
    hf_models = {
        "t5_small_mix": "daanleiva/paraphrase-t5-small-mix",
        "t5_small_qqp": "daanleiva/paraphrase-t5-small-qqp",
        "t5_small_mrpc": "daanleiva/paraphrase-t5-small-mrpc",
        "t5_small_paws": "daanleiva/paraphrase-t5-small-paws",
    }

    labels = list(hf_models.keys())
    st.sidebar.header("Model (Hugging Face)")
    selected_label = st.sidebar.selectbox("Checkpoint", labels, index=0)
    model_path = hf_models[selected_label]
    st.sidebar.info(f"Using Hugging Face model: {model_path}")
else:
    # Use local checkpoints when developing
    models = available_models()
    labels = [lbl for lbl, _ in models]
    paths = {lbl: p for lbl, p in models}

    st.sidebar.header("Model (Local)")
    if not labels:
        st.error("No local checkpoints found.")
        st.stop()
    default_label = "t5_small_mix" if "t5_small_mix" in labels else labels[0]
    selected_label = st.sidebar.selectbox(
        "Checkpoint", labels, index=labels.index(default_label)
    )
    model_path = paths[selected_label]


para = load_paraphraser(model_path)
st.sidebar.success(f"Loaded model from: {model_path}")

st.title("Paraphrase Demo")

# Inputs
txt = st.text_area(
    "Input text",
    "I will follow up later today once I review the latest results from the test run.",
    height=120,
)

colA, colB, colC = st.columns(3)
with colA:
    mode = st.selectbox("Decoding", ["Beam (deterministic)", "Sampling (diverse)"])
with colB:
    n_best = st.slider("n_best", 1, 5, 3)
with colC:
    max_new = st.slider("max_new_tokens", 8, 80, 48, step=4)

# Decoding overrides
if mode.startswith("Beam"):
    num_beams = st.slider("num_beams", 1, 8, max(3, n_best))
    no_repeat = st.slider("no_repeat_ngram_size", 0, 5, 3)
    rep_pen = st.slider("repetition_penalty", 1.0, 1.3, 1.10, step=0.01)
    overrides = dict(
        num_beams=num_beams,
        num_return_sequences=n_best,
        max_new_tokens=max_new,
        no_repeat_ngram_size=no_repeat,
        repetition_penalty=rep_pen,
        do_sample=False,
    )
else:
    temperature = st.slider("temperature", 0.1, 1.5, 0.9, step=0.05)
    top_p = st.slider("top_p", 0.1, 1.0, 0.9, step=0.05)
    top_k = st.slider("top_k", 0, 100, 40, step=5)
    overrides = dict(
        do_sample=True,
        num_beams=1,  # pure sampling to satisfy validation
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=n_best,
        max_new_tokens=max_new,
    )

go = st.button("Paraphrase", type="primary", use_container_width=True)

# Generate
if go and txt.strip():
    outs = para.generate([txt], overrides=overrides)
    st.subheader("Outputs")
    for i, o in enumerate(outs, 1):
        st.markdown(f"**{i}.** {o}")
