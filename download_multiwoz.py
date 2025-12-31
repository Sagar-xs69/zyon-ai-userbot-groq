from datasets import load_dataset
import json

# Download MultiWOZ - task-oriented conversations
dataset = load_dataset("GEM/multi_woz_v22", split="train[:2000]")

task_oriented = []
for item in dataset:
    task_oriented.append({
        "dialogue": item.get("dialogue", []),
        "domain": item.get("domains", []),
        "type": "task_oriented"
    })

with open("multiwoz_tasks.json", "w") as f:
    json.dump(task_oriented, f, indent=2)

print(f"✓ Downloaded {len(task_oriented)} task-oriented conversations!")
