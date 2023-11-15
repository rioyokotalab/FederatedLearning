import hashlib
import json
from transformers.data.data_collator import DataCollatorWithPadding


def get_dataset_digest(dataset):
    s = json.dumps(
        dataset.to_dict() if not type(dataset) is dict else dataset, sort_keys=True
    ).encode("utf-8")
    md5 = hashlib.md5(s).hexdigest()
    return md5


def remove_feature_str(features):
    # for dict item in features, remove key with value type str
    # self.tokenizer.pad will raise exception otherwise
    cache = {}
    for f in features:
        rm_keys = []
        for k, v in f.items():
            if type(v) is str:
                if k not in cache:
                    cache[k] = []
                cache[k].append(v)
                rm_keys.append(k)
        for k in rm_keys:
            f.pop(k)
    return cache


def add_cached_feature_to_batch(batch, cache):
    for k, v in cache.items():
        batch[k] = v


class DataCollatorWithPaddingExcludeStr(DataCollatorWithPadding):
    def __call__(self, features):
        cache = remove_feature_str(features)

        if "labels" in features[0] and type(features[0]["labels"]) is list:
            labels = [x["labels"] for x in features]
            max_t = max([len(x) for x in labels])
            # is seq2seq, pad with -100
            for i, feat in enumerate(features):
                feat["labels"] += [-100] * (max_t - len(feat["labels"]))

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        add_cached_feature_to_batch(batch, cache)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return batch
