# 🎭 Project Bard — Production-Grade MoE LLM Pipeline

<p align="center">
  <em>An end-to-end, massively optimized ~880M parameter Mixture of Experts (MoE) model. Built from scratch and engineered to train directly on a single 16GB Tesla T4 GPU. Features a full MLOps pipeline, Weights & Biases telemetry, and a beautiful realtime Web Dashboard.</em>
</p>

---

## ✨ Enterprise & Scale Features

- **Massive MoE Architecture**: Upgraded to a ~880M Parameter Mixture of Experts (MoE) network with 4 experts (routing to Top-2). It achieves the representational power of a billion-parameter model while keeping active parameters low for fast inference.
- **Extreme VRAM Optimization**: Hand-tuned to train flawlessly on a standard 16GB Tesla T4 GPU.
  - **8-bit AdamW**: Integrates `bitsandbytes` 8-bit AdamW optimizer to slash the optimizer state memory footprint by 75%.
  - **CUDA Memory Management**: Utilizes `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` to completely eliminate memory fragmentation crashes.
  - **Gradient Accumulation**: Uses a micro-batch size of 2 with 32 accumulation steps to maintain a massive global batch size of 64 without triggering PyTorch forward-pass memory spikes.
- **Robust Data Pipeline**: Memory-safe BPE Tokenization that processes data in localized chunks, allowing you to tokenize gigabytes of text without ever exceeding RAM limits.
- **Real-Time MLOps Web Dashboard**:
  - A stunning glassmorphism dark-mode UI with live telemetry.
  - Interactive `Chart.js` graphs streaming **Training Loss**, **Learning Rate** (Cosine Warmup), **GPU Memory (GB)**, and **Tokens/sec** at 1-second intervals.
  - **Dataset Explorer** to preview and audit raw/cleaned text chunks directly in your browser.
- **Cloud Telemetry**: Native Weights & Biases (W&B) integration with automated `online` syncing.

---

## 📁 Project Structure

```text
project_bard/
├── config.py           # Hyperparameters, hardware constraints & MLOps rules
├── data_pipeline.py    # Phase 1: Download, clean, dedup, PII scrub
├── tokenizer.py        # Phase 2: Memory-safe chunked BPE tokenization
├── dataset.py          # Phase 3: Train/val/test split + DataLoaders
├── model.py            # Phase 4: MoE Transformer, GQA, RoPE, RMSNorm, SwiGLU
├── train.py            # Phase 5: 8-bit AdamW + Cosine LR + W&B Telemetry
├── evaluate.py         # Phase 6: Test eval, SFT, DPO alignment
├── generate.py         # CLI inference with streaming
├── chat.py             # Interactive CLI chat interface
├── api_server.py       # FastAPI REST API server + Live Metrics endpoint
├── web/
│   └── index.html      # Glassmorphism Web Chat UI & MLOps Dashboard
├── manage.py           # Unified CLI interface (training, serving, APIs)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🚀 Quick Start

### 1. Installation
Install all dependencies, including PyTorch and the `bitsandbytes` quantization library.
```bash
pip install -r requirements.txt
```

### 2. Full Training Pipeline
You can run the entire pipeline end-to-end using our management script.

```bash
# 1. Scrape, clean, and deduplicate the raw data
./manage.py data

# 2. Train the BPE tokenizer and digitize the dataset into .bin files
./manage.py tokenize

# 3. Launch the 880M Parameter MoE Training Loop
./manage.py train
```
*Note: The training loop automatically handles checkpoint resumption. If you hit `Ctrl+C`, it saves `checkpoints/last.pt`. The next time you run it, it seamlessly resumes your W&B run and optimizer states!*

### 3. MLOps Web Dashboard & Inference
While the model is training, open a new terminal tab and start the API server:
```bash
./manage.py serve
```
- Open your browser to `http://localhost:8000` to access the **Playground UI**.
- Click on the **MLOps Dashboard** tab to watch your training loss and GPU memory curves draw in real-time as the model converges!
- Click on the **Dataset Explorer** to inspect the randomized token streams.

---

## 🏗️ Architecture & Mathematics

This model is built completely from scratch using the most advanced architectural techniques in the open-source LLM space:
- **Mixture of Experts (MoE)**: Instead of a dense MLP, each token is routed to the Top 2 out of 4 specialized feed-forward networks, drastically increasing parameter count without scaling compute time.
- **Grouped-Query Attention (GQA)**: Reduces the Key/Value heads to compress the KV cache, allowing for much larger batch sizes and context windows during inference.
- **Rotary Position Embeddings (RoPE)**: Relative positional encoding that naturally extrapolates to longer context windows.
- **SwiGLU & RMSNorm**: Replaces standard ReLU and LayerNorm for significantly faster convergence and stability.

---

## 📊 W&B Integration
Project Bard is permanently hooked into Weights and Biases. By default, `WANDB_MODE` is forced to `online` in `config.py`. Simply log in once via `wandb login` and your entire training run—including system metrics, gradient norms, and evaluation losses—will securely stream to your cloud dashboard.

## 📝 License

All training data is sourced from public domain works (Project Gutenberg). This codebase is open for experimentation and scale.