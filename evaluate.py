"""
evaluate.py - Phase 6: Comprehensive Evaluation & Alignment
Features:
  - Final test-set perplexity & detailed metrics
  - Multi-genre Supervised Fine-Tuning (SFT) with robust data handling
  - Multi-genre Direct Preference Optimization (DPO) pipeline
  - Post-alignment generation testing across all datasets
  - Configurable execution via CLI arguments (skip specific phases)
  - Memory management (cache clearing) to prevent OOM during long runs
"""
import argparse
import json
import math
import gc
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from config import (
    DEVICE, DTYPE, CHECKPOINT_DIR, VOCAB_SIZE, BLOCK_SIZE,
    SFT_DATA_PATH, DPO_DATA_PATH, SFT_EPOCHS, SFT_LR, DPO_LR, DPO_BETA,
    BATCH_SIZE
)
from model import ShakespeareGPT, ModelConfig
from dataset import get_dataloader
from tokenizer import load_tokenizer


def load_model(checkpoint: str = "best.pt") -> tuple:
    """Load model and config from a checkpoint."""
    ckpt_path = CHECKPOINT_DIR / checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: ModelConfig = ckpt["config"]
    model = ShakespeareGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


# ==============================================================================
# 1. FINAL TEST EVALUATION
# ==============================================================================
@torch.no_grad()
def final_test_evaluation():
    print("=" * 70)
    print("[PHASE 6] Final Test Evaluation")
    print("=" * 70)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if DTYPE == "float16" else torch.bfloat16

    try:
        model, _ = load_model("best.pt")
        model = model.to(device)
    except FileNotFoundError as e:
        print(f"[!] {e}. Skipping evaluation.")
        return

    dl = get_dataloader("test", batch_size=BATCH_SIZE, shuffle=False)
    losses = []
    
    model.eval()
    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            out = model(x, y)
            # Handle both dict (new) and tuple (legacy) return types
            loss = out["loss"] if isinstance(out, dict) else out[1]
        losses.append(loss.item())
    
    # Clear memory after evaluation to prevent OOM in subsequent phases
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(min(mean_loss, 20))  # Cap to prevent overflow
    
    print(f"[+] Test loss: {mean_loss:.4f}")
    print(f"[+] Test perplexity: {perplexity:.2f}")
    print("[+] Base model evaluation complete.\n")


# ==============================================================================
# 2. SUPERVISED FINE-TUNING (SFT) - Multi-Genre
# ==============================================================================
class SFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer, block_size: int = BLOCK_SIZE):
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size
        pad_token_id = tokenizer.token_to_id("[PAD]")
        
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                # Format: [BOS] User: {prompt} \n Assistant: {completion} [EOS]
                text = f"[BOS]User: {obj['prompt'].strip()}\nAssistant: {obj['completion'].strip()}[EOS]"
                ids = tokenizer.encode(text).ids
                if len(ids) > block_size + 1:
                    ids = ids[: block_size + 1]
                self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        pad_len = self.block_size + 1 - len(ids)
        pad_token_id = self.tokenizer.token_to_id("[PAD]")
        ids = ids + [pad_token_id] * pad_len
        
        t = torch.tensor(ids, dtype=torch.long)
        x = t[:-1]
        y = t[1:].clone()
        
        # Mask loss on padding tokens
        y[x == pad_token_id] = -100
        return x, y


def create_comprehensive_sft():
    """Generates a rich, multi-genre SFT dataset covering all 6 books."""
    samples = [
        # Shakespeare
        {"prompt": "Who is the tragic prince of Denmark?", "completion": "The tragic prince of Denmark is Hamlet, who seeks to avenge his father's murder by his uncle, King Claudius."},
        {"prompt": "What is the famous opening line of Richard III?", "completion": "Now is the winter of our discontent made glorious summer by this sun of York."},
        {"prompt": "Describe the relationship between Romeo and Juliet.", "completion": "Romeo and Juliet are star-crossed lovers from feuding families, the Montagues and the Capulets, whose tragic romance ends in their untimely deaths."},
        # Sherlock Holmes
        {"prompt": "What is the significance of the mud on the boot?", "completion": "The peculiar reddish mud indicates the suspect recently traveled from the specific clay-rich soils of the Sussex countryside."},
        {"prompt": "How does Sherlock Holmes typically solve his cases?", "completion": "Holmes relies on deductive reasoning, meticulous observation of trivial details, and a profound knowledge of criminal history."},
        {"prompt": "Who is Sherlock Holmes's loyal companion?", "completion": "Dr. John Watson, a former army doctor, serves as Holmes's friend, biographer, and steadfast companion in their investigations."},
        # Pride & Prejudice
        {"prompt": "Why is Mr. Darcy initially disliked by Elizabeth Bennet?", "completion": "Mr. Darcy is initially perceived as proud, arrogant, and disdainful of the local society, particularly after he slights Elizabeth at a ball."},
        {"prompt": "What is the central theme of Pride and Prejudice?", "completion": "The novel explores the delicate balance between love, reputation, and class in 19th-century English society."},
        {"prompt": "Who is the witty and observant protagonist of the novel?", "completion": "Elizabeth Bennet, the second of the five Bennet daughters, is known for her sharp wit, lively mind, and initial prejudice against Mr. Darcy."},
        # Frankenstein
        {"prompt": "What drives Victor Frankenstein to create life?", "completion": "Victor is driven by an obsessive, hubristic desire to unlock the secrets of life and death, hoping to banish disease and achieve glory."},
        {"prompt": "How does the creature react to his rejection by society?", "completion": "Initially benevolent, the creature becomes embittered and vengeful after being repeatedly rejected and attacked by humans due to his grotesque appearance."},
        {"prompt": "Where does Victor Frankenstein pursue his studies?", "completion": "Victor pursues his natural philosophy and alchemical studies at the University of Ingolstadt in Germany."},
        # Alice in Wonderland
        {"prompt": "How does Alice enter Wonderland?", "completion": "Alice follows a hurried White Rabbit down a large rabbit hole, falling for a considerable time before landing in the fantastical realm."},
        {"prompt": "What is the Cheshire Cat known for?", "completion": "The Cheshire Cat is known for its distinctive, mischievous grin and its ability to appear and disappear at will, often offering cryptic advice."},
        {"prompt": "Who rules Wonderland with an iron fist?", "completion": "The Queen of Hearts, a foul-tempered monarch who frequently orders the beheading of those who displease her."},
        # Don Quixote
        {"prompt": "Who is Don Quixote's loyal squire?", "completion": "Sancho Panza, a pragmatic and earthy peasant, who accompanies his master on his delusional chivalric quests in hopes of governing an island."},
        {"prompt": "What does Don Quixote famously attack, believing them to be giants?", "completion": "Don Quixote famously attacks windmills, mistaking them for monstrous giants in a display of his chivalric delusion."},
        {"prompt": "What is the name of Don Quixote's horse?", "completion": "His horse is named Rocinante, a former plow horse that Quixote believes to be a noble and powerful steed."},
    ]
    with SFT_DATA_PATH.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"[+] Generated comprehensive SFT dataset with {len(samples)} multi-genre examples.")


def run_sft():
    print("=" * 70)
    print("[PHASE 6] Supervised Fine-Tuning (SFT)")
    print("=" * 70)

    if not SFT_DATA_PATH.exists():
        print("[!] SFT data not found; generating comprehensive multi-genre sample...")
        create_comprehensive_sft()

    tokenizer = load_tokenizer()
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    try:
        base_ckpt_path = CHECKPOINT_DIR / "best.pt"
        base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
        cfg: ModelConfig = base_ckpt["config"]
    except FileNotFoundError:
        print("[!] Base model 'best.pt' not found. Cannot run SFT.")
        return
    
    model = ShakespeareGPT(cfg).to(device)
    model.load_state_dict(base_ckpt["model_state_dict"])
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=SFT_LR, weight_decay=0.01)

    ds = SFTDataset(SFT_DATA_PATH, tokenizer)
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    print(f"[*] Starting SFT for {SFT_EPOCHS} epochs on {len(ds)} examples...")
    for epoch in range(SFT_EPOCHS):
        total, n = 0.0, 0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(x, y)
            loss = out["loss"] if isinstance(out, dict) else out[1]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n += 1
        print(f"    [SFT epoch {epoch+1}/{SFT_EPOCHS}] loss={total/max(1,n):.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }, CHECKPOINT_DIR / "sft_model.pt")
    print("[+] SFT model saved successfully.\n")


# ==============================================================================
# 3. DIRECT PREFERENCE OPTIMIZATION (DPO) - Multi-Genre
# ==============================================================================
def create_comprehensive_dpo():
    """Generates a rich, multi-genre DPO dataset (Chosen vs. Rejected)."""
    samples = [
        {
            "prompt": "Describe the atmosphere of the old mansion.",
            "chosen": "The mansion stood in brooding silence, its ivy-choked walls and shattered windows whispering tales of a long-forgotten, tragic past.",
            "rejected": "The house was old and had lots of plants on it and broken windows and it was quiet."
        },
        {
            "prompt": "Explain the detective's deduction about the cigar ash.",
            "chosen": "The ash is grey and flaky, characteristic only of a Trichinopoly cigar, which immediately narrows our suspect to someone with specific colonial ties.",
            "rejected": "The detective looked at the ash and knew it was from a cigar and that the person was from a far away place."
        },
        {
            "prompt": "Summarize the creature's plea to Victor.",
            "chosen": "The creature eloquently demands that Victor create a female companion for him, promising to vanish into the wilderness if his request is granted.",
            "rejected": "The monster told Victor to make him a wife so he would go away and not bother anyone anymore."
        },
        {
            "prompt": "Describe Alice's feelings upon shrinking.",
            "chosen": "A wave of profound bewilderment washed over her, as the familiar contours of the room expanded into a vast, intimidating landscape.",
            "rejected": "Alice felt weird and confused because everything was getting really big and she was small."
        },
        {
            "prompt": "How does Sancho Panza view his master's adventures?",
            "chosen": "Sancho views the adventures with a mixture of bewildered loyalty and pragmatic skepticism, often pointing out the mundane reality behind his master's grand illusions.",
            "rejected": "Sancho thought his master was crazy but followed him anyway because he wanted to get rich."
        },
        {
            "prompt": "What is the nature of Mr. Darcy's first proposal to Elizabeth?",
            "chosen": "His proposal is a masterful blend of ardent passion and unintentional insult, as he confesses his love while detailing the social degradation he feels in marrying her.",
            "rejected": "Mr. Darcy asked Elizabeth to marry him but he was kind of mean about it and said she was poor."
        }
    ]
    with DPO_DATA_PATH.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"[+] Generated comprehensive DPO dataset with {len(samples)} preference pairs.")


def run_dpo():
    print("=" * 70)
    print("[PHASE 6] Direct Preference Optimization (DPO)")
    print("=" * 70)

    try:
        from trl import DPOTrainer, DPOConfig
        from datasets import Dataset as HFDataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[!] trl/transformers not installed. Skipping DPO. Install with: pip install trl transformers datasets")
        return

    if not DPO_DATA_PATH.exists():
        print("[!] DPO data not found; generating comprehensive multi-genre sample...")
        create_comprehensive_dpo()

    print("[*] Loading reference model for DPO (stand-in HF model)...")
    model_name = "sshleifer/tiny-gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = []
    with DPO_DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append({
                "prompt": obj["prompt"],
                "chosen": obj["chosen"],
                "rejected": obj["rejected"],
            })
    hf_ds = HFDataset.from_list(data)

    training_args = DPOConfig(
        output_dir=str(CHECKPOINT_DIR / "dpo"),
        learning_rate=DPO_LR,
        beta=DPO_BETA,
        per_device_train_batch_size=2,
        max_steps=30,
        logging_steps=10,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=hf_ds,
        processing_class=tokenizer,
    )
    
    print("[*] Starting DPO training...")
    trainer.train()
    trainer.save_model(str(CHECKPOINT_DIR / "dpo_final"))
    print("[+] DPO training complete.\n")


# ==============================================================================
# 4. POST-ALIGNMENT GENERATION TEST
# ==============================================================================
def test_generation_post_alignment():
    print("=" * 70)
    print("[PHASE 6] Post-Alignment Generation Test")
    print("=" * 70)
    
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    
    ckpt_path = CHECKPOINT_DIR / "sft_model.pt"
    if not ckpt_path.exists():
        print("[!] SFT model not found. Skipping generation test.")
        return
        
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: ModelConfig = ckpt["config"]
    model = ShakespeareGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    
    tokenizer = load_tokenizer()
    
    test_prompts = [
        ("Shakespeare", "User: Who is the tragic prince of Denmark?\nAssistant:"),
        ("Sherlock", "User: What is the significance of the mud on the boot?\nAssistant:"),
        ("Austen", "User: Why is Mr. Darcy initially disliked?\nAssistant:"),
        ("Shelley", "User: What drives Victor Frankenstein to create life?\nAssistant:"),
        ("Carroll", "User: How does Alice enter Wonderland?\nAssistant:"),
        ("Cervantes", "User: Who is Don Quixote's loyal squire?\nAssistant:"),
    ]
    
    print("[*] Generating responses for each genre...\n")
    for genre, prompt in test_prompts:
        print(f"--- {genre} ---")
        ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        
        with torch.no_grad():
            out_ids = model.generate(
                idx, 
                max_new_tokens=60, 
                temperature=0.7, 
                top_k=40, 
                top_p=0.9, 
                repetition_penalty=1.2
            )
        
        new_text = tokenizer.decode(out_ids[0][len(ids):].tolist()).strip()
        print(f"Prompt: {prompt}")
        print(f"Output: {new_text}\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Evaluation and Alignment")
    parser.add_argument("--skip-test", action="store_true", help="Skip final test evaluation")
    parser.add_argument("--skip-sft", action="store_true", help="Skip Supervised Fine-Tuning")
    parser.add_argument("--skip-dpo", action="store_true", help="Skip Direct Preference Optimization")
    parser.add_argument("--skip-gen", action="store_true", help="Skip post-alignment generation test")
    args = parser.parse_args()

    if not args.skip_test:
        final_test_evaluation()
    
    if not args.skip_sft:
        run_sft()
        
    if not args.skip_dpo:
        run_dpo()
        
    if not args.skip_gen:
        test_generation_post_alignment()


if __name__ == "__main__":
    main()