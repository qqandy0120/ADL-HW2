import json
from pathlib import Path
import os
import pandas as pd
ids, answers = [], []

with open(os.path.join("ckpt", "qa", "infer", "predict_predictions.json"), "r", encoding="utf-8") as f:
    data = json.load(f)
    for id, ans in data.items():
        ids.append(id)
        answers.append(ans)

df = pd.DataFrame({
    "id": ids,
    "answer": answers
})

df.to_csv("prediction.csv", index=False, encoding="utf-8")

