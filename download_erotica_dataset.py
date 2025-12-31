from datasets import load_dataset
import json

print("⚠️  WARNING: This is explicit adult content")
print("⚠️  Will NOT work with Gemini API (responses blocked)")
print("⚠️  May violate Telegram ToS (account ban risk)")
print("\nDownloading...")

try:
    # Download erotica-analysis dataset
    dataset = load_dataset("openerotica/erotica-analysis", split="train[:1000]")
    
    erotica_data = []
    for item in dataset:
        erotica_data.append({
            "title": item.get("title", ""),
            "summary": item.get("summary", ""),
            "tags": item.get("tags", []),
            "prompt": item.get("prompt", ""),
            "type": "adult_content"
        })
    
    with open("erotica_analysis.json", "w") as f:
        json.dump(erotica_data, f, indent=2)
    
    print(f"\n✓ Downloaded {len(erotica_data)} erotica examples")
    print("\n⚠️  REMINDER: Gemini API will refuse to generate explicit content")
    print("⚠️  This data is for research/analysis only with your current setup")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nNote: Dataset may require authentication or have access restrictions")
