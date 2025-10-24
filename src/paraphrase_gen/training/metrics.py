import evaluate

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


def compute_metrics(eval_preds):
    """
    Computes BLEU and ROUGE scores for predictions and references.
    Expects eval_preds as a tuple of lists of strings
    """
    preds, refs = eval_preds
    preds = [p.strip() for p in preds]
    refs = [[r.strip()] for r in refs]
    b = bleu.compute(predictions=[[x] for x in preds], references=[[y] for y in refs])
    r = rouge.compute(predictions=preds, references=refs)
    return {
        "bleu": b.get("bleu", 0.0),
        "rouge1": r.get("rouge1", 0.0),
        "rouge2": float(r.get("rouge2", 0.0)) if "rouge2" in r else 0.0,
        "rougeL": float(r.get("rougeL", 0.0)),
        "rougeLsum": float(r.get("rougeLsum", 0.0)) if "rougeLsum" in r else 0.0,
    }
