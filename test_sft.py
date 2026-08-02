import torch
from model import ShakespeareGPT, ModelConfig
from tokenizer import load_tokenizer

tokenizer = load_tokenizer()
ckpt = torch.load("checkpoints/sft_model.pt", map_location="cuda", weights_only=False)
base_ckpt = torch.load("checkpoints/best.pt", map_location="cuda", weights_only=False)
model = ShakespeareGPT(base_ckpt["config"]).cuda()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

prompt = "System: You are Bard.\n\nUser: to be or not to be\n\nBard: "
ids = tokenizer.encode(prompt).ids
idx = torch.tensor([ids], dtype=torch.long, device="cuda")

with torch.no_grad():
    generated_ids = model.generate(idx, max_new_tokens=50, temperature=0.8)
    
print(tokenizer.decode(generated_ids[0].tolist()))
