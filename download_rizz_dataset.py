from datasets import load_dataset
import json

# Download the rizz corpus
dataset = load_dataset("the-rizz/the-rizz-corpus")

# Save to JSON
with open("rizz_conversations.json", "w") as f:
    json.dump(dataset['train'].to_dict(), f, indent=2)

print("✓ Rizz dataset downloaded!")
