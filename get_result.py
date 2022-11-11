import json
from pathlib import Path
import os
import pandas as pd
import json
import typing
from typing import List, Dict
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import seed
ids, answers = [], []


def main(args):
    with open(os.path.join("ckpt", "qa", "infer", "predict_predictions.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
        for id, ans in data.items():
            ids.append(id)
            answers.append(ans)

    df = pd.DataFrame({
        "id": ids,
        "answer": answers
    })

    df.to_csv(args.pred_path, index=False, encoding="utf-8")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--pred_path",
        type=Path,
        help="Directory to the final prediction csv.",
        default="./prediction.csv",
    )

    return args

if __name__ == '__main__':
    args = parse_args()
    args.pred_path.mkdir(parents=True, exist_ok=True)
    main(args)