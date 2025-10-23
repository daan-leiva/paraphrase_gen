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
    refs = [r.strip() for r in refs]
    b = bleu.compute(predictions=[[x] for x in preds], references=[[y] for y in refs])
    r = rouge.compute(predictions=preds, references=refs)
    return {
        "bleu": b.get("bleu", 0.0),
        "rouge1": r.get("rouge1", 0.0),
        "rougeL": r.get("rougeL", 0.0),
    }
