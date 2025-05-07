import json
from datasets import load_dataset

# Register your custom dataset class
from logiqa import LogiQA  # assuming your class is saved in `logiqa_dataset.py`

# Load the dataset using your builder
dataset = load_dataset("logiqa.py",trust_remote_code=True)  # or change path if needed

# Save each split to JSONL
for split_name in dataset:
    with open(f"logiqa_{split_name}.jsonl", "w", encoding="utf-8") as f:
        for example in dataset[split_name]:
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")
