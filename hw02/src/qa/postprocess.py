from transformers import EvalPrediction

from src.qa.utils import postprocess_qa_predictions

answer_column_name = "answers"

def post_processing_function(
    examples,
    features,
    predictions,
    version_2_with_negative,
    n_best_size,
    max_answer_length,
    null_score_diff_threshold,
    output_dir,
    log_level,
    stage="eval"
):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=version_2_with_negative,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
        null_score_diff_threshold=null_score_diff_threshold,
        output_dir=output_dir,
        log_level=log_level,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples] if answer_column_name in examples[0] else None
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)
