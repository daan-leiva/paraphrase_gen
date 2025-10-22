# Paraphrase Generation System

A machine learning pipeline end-to-end pipeline for generating paraphrases of text using sequence-to-sequence models.
This project fine tunes a Transformer model and exposes a lightweight REST API to generate alternative formulations of input sequences.

## Key Features

- Fine-tunes a pre-trained sequence-to-sequence model (TODO: Model Names here) on paraphrase pairs.
- Provides modular components for training, inference, evaluation, and deployment
- Includes a FastAPI server to serve model predictions via '/paraphrase' endpoint.
- Supports automatic metrics (BLEU, ROUGE) and standarsized workflows.

## Quick Start
```bash
git clone https://github.com/daan-leiva/paraphrase_gen.git  
cd paraphrase_gen  
