from datasets import load_dataset
import json

# Download SmolTalk - Use 'all' config for complete dataset
dataset = load_dataset("HuggingFaceTB/smoltalk", "all", split="train[:10000]")

# Save subset
with open("mature_conversations.json", "w") as f:
    json.dump([{
        "messages": item["messages"]
    } for item in dataset], f, indent=2)

print("✓ Mature conversation dataset downloaded!")
