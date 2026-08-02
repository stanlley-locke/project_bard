import json
from datasets import load_dataset
from pathlib import Path

def download_sft_data(output_path, num_examples=5000):
    print("Downloading databricks-dolly-15k dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    data = []
    for i, row in enumerate(dataset):
        if i >= num_examples:
            break
            
        system_prompt = "You are Bard, a helpful and knowledgeable AI assistant."
        # Combine instruction and context for the prompt
        prompt = row['instruction']
        if row['context']:
            prompt += f"\nContext: {row['context']}"
            
        data.append({
            "system": system_prompt,
            "prompt": prompt,
            "response": row['response'],
            "context": ""
        })
        
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_file, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")
            
    print(f"Saved {len(data)} high-quality SFT examples to {output_path}")

if __name__ == "__main__":
    download_sft_data("data/sft_shakespeare.jsonl", 5000)
