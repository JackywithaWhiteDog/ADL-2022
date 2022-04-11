from datasets import load_metric
from transformers import EvalPrediction

def compute_metrics(p: EvalPrediction, version_2_with_negative):
    metric = load_metric("squad_v2" if version_2_with_negative else "squad")
    return metric.compute(predictions=p.predictions, references=p.label_ids)
