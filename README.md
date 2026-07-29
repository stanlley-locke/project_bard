# Project Bard — Shakespeare LLM (Full Pipeline)

An end-to-end implementation of a decoder-only Transformer trained on the
Complete Works of Shakespeare, built to mirror the 6-phase enterprise LLM
pipeline (data curation, tokenization, structured batching, modern
architecture, optimized pre-training, and alignment).

## Project Structure
- `config.py` — hyperparameters & paths
- `data_pipeline.py` — Phase 1: download, clean, dedup, PII scrub
- `tokenizer.py` — Phase 2: BPE tokenizer + memmap digitization
- `dataset.py` — Phase 3: train/val/test split + PyTorch DataLoaders
- `model.py` — Phase 4: Transformer with RMSNorm, RoPE, MHA
- `train.py` — Phase 5: AdamW + cosine LR + grad clip + bfloat16
- `evaluate.py` — Phase 6: test eval, SFT, DPO
- `generate.py` — inference

## Quick Start
```bash
pip install -r requirements.txt
python data_pipeline.py      # Phase 1
python tokenizer.py          # Phase 2
python dataset.py            # Phase 3
python train.py              # Phase 4 + 5 (architecture + pre-training)
python evaluate.py           # Phase 6
python generate.py           # sample output