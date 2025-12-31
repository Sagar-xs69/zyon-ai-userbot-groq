from datasets import load_dataset
import json

# Download UltraChat 200K - state of the art conversations
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:5000]")

# Save subset
conversations = []
for item in dataset:
    conversations.append({
        "messages": item["messages"],
        "type": "quality_conversation"
    })

with open("ultrachat_conversations.json", "w") as f:
    json.dump(conversations, f, indent=2)

print(f"✓ Downloaded {len(conversations)} ultra-quality conversations!")
