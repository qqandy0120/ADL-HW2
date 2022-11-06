import json
import typing
from typing import List, Dict
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import seed

SPLIT = ['train', 'test', 'valid']

def main(args):
    seed(args.rand_seed)
    # context_path = os.path.join(args.data_dir, 'context.json')
    context_path: Path = args.data_dir / "context.json"
    context: List = json.loads(context_path.read_text())
    id2context = {i: j for i, j in enumerate(context)}
    id2context_path: Path = args.output_dir / "id2context.json"
    id2context_path.write_text(json.dumps(id2context, ensure_ascii=False))

    data_paths: Dict[str, Path] = {split: args.data_dir / f"{split}.json" for split in SPLIT}

    swagformat_data = {
        split: [{
                "split": split,
                "id": instance["id"],
                "sent1": instance["question"],
                "sent2": "",
                "gold_source": None,
                "paragraph1": id2context[instance["paragraphs"][0]],
                "paragraph2": id2context[instance["paragraphs"][1]],
                "paragraph3": id2context[instance["paragraphs"][2]],
                "paragraph4": id2context[instance["paragraphs"][3]],
                "label": instance["paragraphs"].index(instance["relevant"])+1 if split in ["train", "valid"] else None
            } for instance in json.loads(path.read_text())]
        for split, path in data_paths.items()
    }

    (args.output_dir / "swagformat_data").mkdir(parents=True, exist_ok=True)
    swagformat_paths: Dict[str, Path] = {split: args.output_dir / "swagformat_data" / f"{split}.json" for split in SPLIT}

    for split in SPLIT:
        swagformat_paths[split].write_text(json.dumps(swagformat_data[split],indent=2, ensure_ascii=False))


    


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
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./cache",
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)