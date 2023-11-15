import os
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from evaluate import load

from .data_utils import DataCollatorWithPaddingExcludeStr, get_dataset_digest

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

key_met_map = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "accuracy",
    "stsb": "pearson",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "qqp": "accuracy",
    "rte": "accuracy",
    "wnli": "accuracy",
}

class GLUEDataManager:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.collate_fn = DataCollatorWithPaddingExcludeStr(self.tokenizer)
    
    def load_dataset(self, dataset_name):
        # Load the dataset
        # Set HF_DATASETS_CACHE environment variable (eg. ~/.cache/huggingface/datasets) to use cache
        raw_datasets = load_dataset("glue", dataset_name)

        if dataset_name == "stsb":
            self.num_labels = 1
            label2id = None
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            self.num_labels = len(label_list)
            label2id = {v: i for i, v in enumerate(label_list)}
        
        max_seq_length = self.config.max_seq_length
        sentence1_key, sentence2_key = task_to_keys[dataset_name]

        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],)
                if sentence2_key is None
                else (examples[sentence1_key], examples[sentence2_key])
            )
            result = self.tokenizer(*args, max_length=max_seq_length, truncation=True)

            # Map labels to IDs (not necessary for GLUE tasks)
            if label2id is not None and "label" in examples:
                result["label"] = [(label2id[lbl] if lbl != -1 else -1) for lbl in examples["label"]]
            return result

        arr_cache_names = None
        if os.getenv("HF_DATASETS_CACHE", None) is not None:
            arr_cache_names = {
                k: os.path.join(
                    os.getenv("HF_DATASETS_CACHE"),
                    "{}-{}-{}".format(
                        k, self.tokenizer.name_or_path.split("/")[-1], get_dataset_digest(v)
                    ),
                )
                for k, v in raw_datasets.items()
            }

        raw_datasets = raw_datasets.map(
            preprocess_function, batched=True, cache_file_names=arr_cache_names
        )
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["validation_matched" if dataset_name == "mnli" else "validation"]
        test_dataset = None

        return train_dataset, eval_dataset, test_dataset
    
    def compute_metrics(self, dataset_name, preds, labels):
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        metric = load("glue", dataset_name)
        preds = np.squeeze(preds) if dataset_name == "stsb" else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=labels)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        result["key_score"] = result[key_met_map[dataset_name]]
        return result
