from datasets import Dataset, DatasetDict
from typing import List, Dict, Any
import json

from src.config import PARAGRAPHS_NAME

def load_json(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        result = json.load(f)
    return result

def parse_data(data: List[Dict[str, Any]], context: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "id": d["id"],
            "question": d["question"],
            "paragraphs": [
                context[para]
                for para in d["paragraphs"]
            ],
            **({
                "label": d["paragraphs"].index(d["relevant"])
            } if "relevant" in d else {})
        }
        for d in data
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

def get_datasets(data_files: Dict[str, str], context_file: str) -> DatasetDict:
    context = load_json(context_file)
    return DatasetDict({
        split: parse_dataset(parse_data(load_json(data_file), context))
        for split, data_file in data_files.items()
    })
