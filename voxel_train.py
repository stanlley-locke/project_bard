"""
voxel_train.py - Distill trained model weights into voxel space
"""
import torch
from pathlib import Path
from config import DEVICE, CHECKPOINT_DIR
from model import ShakespeareGPT, ModelConfig
from tokenizer import load_tokenizer
from voxel_engine import VoxelEngine

def train_voxel_model():
    print("=" * 70)
    print(" VOXEL TRAINING: Distilling Model Weights to 3D Space")
    print("=" * 70)
    
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    
    # Load trained model
    ckpt_path = CHECKPOINT_DIR / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    print("[*] Loading trained model...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: ModelConfig = ckpt["config"]
    
    model = ShakespeareGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    
    print("[*] Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    # Initialize and train voxel engine
    print("[*] Initializing Voxel Engine...")
    engine = VoxelEngine(grid_size=128, pca_components=3)
    
    print("[*] Distilling model weights into voxel space...")
    engine.train_from_model(model, tokenizer)
    
    # Save trained voxel model
    voxel_path = CHECKPOINT_DIR / "voxel_model"
    engine.save(voxel_path)
    
    # Print statistics
    stats = engine.get_stats()
    print("\n" + "=" * 70)
    print(" VOXEL MODEL STATISTICS")
    print("=" * 70)
    print(f"Grid Size: {stats['grid_size']}x{stats['grid_size']}x{stats['grid_size']}")
    print(f"Total Voxels: {stats['total_voxels']:,}")
    print(f"Occupied Voxels: {stats['occupied_voxels']:,} ({stats['occupied_voxels']/stats['total_voxels']*100:.2f}%)")
    print(f"Total Tokens: {stats['total_tokens']:,}")
    print(f"Memory Footprint: {stats['memory_mb']:.2f} MB")
    print(f"Trajectory Tunnels: {stats['num_tunnels']}")
    print("=" * 70)
    
    print("[+] Voxel training complete!")

if __name__ == "__main__":
    train_voxel_model()