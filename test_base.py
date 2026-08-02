import torch
from model import ShakespeareGPT, ModelConfig
from tokenizer import load_tokenizer

tokenizer = load_tokenizer()
base_ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
model = ShakespeareGPT(base_ckpt["config"])
model.load_state_dict(base_ckpt["model_state_dict"])
model.eval()

prompt = "The following is a conversation between a human and Bard, a knowledgeable and eloquent assistant specialising in the works of William Shakespeare and Elizabethan literature.\n\nHuman: to be or not to be\nBard: "
ids = tokenizer.encode(prompt).ids
idx = torch.tensor([ids], dtype=torch.long)

with torch.no_grad():
    generated_ids = model.generate(idx, max_new_tokens=50, temperature=0.8)
    
print(tokenizer.decode(generated_ids[0].tolist()))
