import json
import typing
from typing import List, Dict, Optional, Union
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import tensorflow as tf
from transformers import BertTokenizerFast
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

SPLIT = ['train', 'test', 'valid']
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
ending_names = ["paragraph1", "paragraph2", "paragraph3", "paragraph4"]

def preprocess_function(examples):
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    question_headers = examples["sent2"]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def main(args):
    swag_paths: Dict[str, Path] = {split: args.swagformat_data_dir / f"{split}.json" for split in SPLIT
    }
    swag = {split: json.loads(path.read_text()) for split, path in swag_paths.items()}

    print(swag["train"][0].keys())

    tokenized_swag = swag.map(preprocess_function, batched=True)

    model = AutoModelForMultipleChoice.from_pretrained("bert-base-chinese")
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_swag["train"],
        eval_dataset=tokenized_swag["eval"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    )

    trainer.train()



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data",
    )
    parser.add_argument("--rand_seed", type=int, help="Random seed.", default=7777)
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./cache",
    )
    parser.add_argument(
        "--swagformat_data_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./cache/swagformat_data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="./results"
    )
    parser.add_argument(
        "--evaluation_startegy",
        type=str,
        default="epoch"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=0.01,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.swagformat_data_dir.mkdir(parents=True, exist_ok=True)
    main(args)