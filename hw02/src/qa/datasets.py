from datasets import Dataset, DatasetDict
from typing import List, Dict, Any, Optional
import json

def load_json(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        result = json.load(f)
    return result

def parse_data(data: List[Dict[str, Any]], context: List[str], relevant_ids: Optional[List[int]]=None) -> List[Dict[str, Any]]:
    if relevant_ids is None:
        return [
            {
                "id": d["id"],
                "question": d["question"],
                "context": context[d["relevant"]],
                **({
                    "answers": {
                        "text": [d["answer"]["text"]],
                        "answer_start": [d["answer"]["start"]]
                    }
                } if "answer" in d else {}) 
            }
            for d in data
        ]
    return [
        {
            "id": d["id"],
            "question": d["question"],
            "context": context[d["paragraphs"][r]],
            **({
                "answers": {
                    "text": [d["answer"]["text"]],
                    "answer_start": [d["answer"]["start"]]
                }
            } if "answer" in d else {})
        }
        for d, r in zip(data, relevant_ids)
    ]

def parse_dataset(data: List[Dict[str, Any]]) -> Dataset:
    result = {
        key: []
        for key in data[0].keys()
    }
    for d in data:
        for key, value in d.items():
            result[key].append(value)
    return Dataset.from_dict(result)

def get_datasets(data_files: Dict[str, str], context_file: str, relevant_ids: Optional[List[int]]=None) -> DatasetDict:
    context = load_json(context_file)
    return DatasetDict({
        split: parse_dataset(parse_data(load_json(data_file), context, relevant_ids))
        for split, data_file in data_files.items()
    })