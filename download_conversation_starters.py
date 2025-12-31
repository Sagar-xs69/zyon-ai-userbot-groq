from datasets import load_dataset
import json

# Download conversation starters
dataset = load_dataset("Langame/conversation-starters")

# Get all starters (they're all good conversation topics)
all_starters = []
for item in dataset['train']:
    all_starters.append({
        "question": item.get('question', ''),
        "context": item.get('context', ''),
        "type": "conversation_starter"
    })

with open("conversation_starters.json", "w") as f:
    json.dump(all_starters, f, indent=2)

print(f"✓ Downloaded {len(all_starters)} conversation starters!")
