from datasets import load_dataset
import json

# Download RealTalk - authentic human conversations
try:
    dataset = load_dataset("ProlificAI/realtalk-dataset", split="train[:3000]")
    
    real_convos = []
    for item in dataset:
        real_convos.append({
            "conversation": item.get("messages", []),
            "context": "real_human_dialogue"
        })
    
    with open("realtalk_authentic.json", "w") as f:
        json.dump(real_convos, f, indent=2)
    
    print(f"✓ Downloaded {len(real_convos)} authentic conversations!")
except Exception as e:
    print(f"Note: RealTalk may require access - {e}")
