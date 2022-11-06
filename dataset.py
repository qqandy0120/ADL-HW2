import typing
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
import re
import torch
import json, pickle
import numpy as np

class ContentSlecDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        label_mapping: Dict[str, str],
        max_len: int,
    ):
        self.data = data
        self.id2context: Dict[int, str] = {int(k): v for k, v in label_mapping.items()}
        self.context2id: Dict[str, int] = {v: int(k) for k, v in label_mapping.items()}
        self.max_len = max_len
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance
    
    def collate_fn(self, samples: List[Dict]) -> Dict:
        ids = [sample['id'] for sample in samples]
        seqs = [{
            'question': sample['question'],
            'paragraph': idx,
        } for sample in samples for idx in sample['paragraphs']]
        relevants = [sample['relevant'] for sample in samples]
        answers = [sample['answer'] for sample in samples]

        return{
            'ids': ids,
            'seqs': seqs,
            'relevants': relevants,
            'answers': answers,
        }


if __name__ == '__main__':

    with open('./data/train.json') as f:
        data = json.load(f)
    with open('./cache/id2context.json') as f:
        label_mapping = json.load(f)
    dataset = ContentSlecDataset(
        data=data,
        label_mapping=label_mapping,
        max_len=100,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=3,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )

    for i, batch in enumerate(dataloader):
        print(batch)
        if i == 3:
            break