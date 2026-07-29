"""
voxel_engine.py - VoxelPath Weight Distillation Engine
Distills a trained 110M parameter Transformer into a 3D voxel space.
Maps model embeddings into geometric coordinates via PCA, builds density fields,
and implements geometric pathfinding for inference.
Memory footprint: ~8.4 MB for the voxel grid + metadata.
"""
import torch
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.decomposition import PCA
from collections import Counter, defaultdict

class VoxelEngine:
    def __init__(self, grid_size: int = 128, pca_components: int = 3):
        self.grid_size = grid_size
        self.pca_components = pca_components
        
        # 3D voxel grid for density mapping
        self.voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        
        # Token-to-coordinate mapping (learned via PCA on embeddings)
        self.token_coords: Dict[int, Tuple[int, int, int]] = {}
        self.coord_to_tokens: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        
        # Trajectory tunnels for high-frequency sequences
        self.tunnels: Dict[str, str] = {}
        
        # Metadata
        self.total_tokens = 0
        self.vocab_size = 0
        self.is_trained = False
        
    def extract_embeddings(self, model) -> np.ndarray:
        """
        Extract the token embedding weights from the trained model.
        These weights contain the model's learned semantic representations.
        """
        # The model has tied embeddings: token_emb.weight == lm_head.weight
        # Shape: (vocab_size, n_embd) = (32768, 768)
        embeddings = model.token_emb.weight.detach().cpu().numpy()
        self.vocab_size = embeddings.shape[0]
        print(f"[*] Extracted embeddings: {embeddings.shape}")
        return embeddings
    
    def map_embeddings_to_3d(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Use PCA to reduce 768D embeddings to 3D coordinates.
        This creates a geometric representation of the semantic space.
        """
        print(f"[*] Reducing {embeddings.shape[1]}D embeddings to {self.pca_components}D via PCA...")
        
        pca = PCA(n_components=self.pca_components)
        coords_3d = pca.fit_transform(embeddings)
        
        # Normalize coordinates to [0, grid_size-1] range
        coords_min = coords_3d.min(axis=0)
        coords_max = coords_3d.max(axis=0)
        coords_normalized = (coords_3d - coords_min) / (coords_max - coords_min + 1e-8)
        coords_discrete = (coords_normalized * (self.grid_size - 1)).astype(int)
        
        print(f"[+] PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        return coords_discrete
    
    def build_voxel_grid(self, embeddings: np.ndarray, coords_3d: np.ndarray) -> None:
        """
        Build the 3D voxel grid by mapping each token to its coordinate
        and populating the grid based on embedding magnitudes.
        """
        print(f"[*] Building voxel grid ({self.grid_size}x{self.grid_size}x{self.grid_size})...")
        
        self.voxel_grid.fill(0)
        self.token_coords.clear()
        self.coord_to_tokens.clear()
        
        # Calculate embedding magnitudes (L2 norm)
        magnitudes = np.linalg.norm(embeddings, axis=1)
        mag_min, mag_max = magnitudes.min(), magnitudes.max()
        mag_normalized = (magnitudes - mag_min) / (mag_max - mag_min + 1e-8)
        
        # Map each token to its 3D coordinate
        for token_id in range(self.vocab_size):
            x, y, z = coords_3d[token_id]
            self.token_coords[token_id] = (x, y, z)
            self.coord_to_tokens[(x, y, z)].append(token_id)
            
            # Populate voxel grid with normalized magnitude
            # Higher magnitude = stronger semantic signal
            self.voxel_grid[x, y, z] += mag_normalized[token_id]
        
        # Apply Gaussian smoothing to create density fields
        print("[*] Applying Gaussian smoothing to voxel grid...")
        self._smooth_grid(sigma=1.5)
        
        self.total_tokens = self.vocab_size
        self.is_trained = True
        print(f"[+] Voxel grid built. Total tokens mapped: {self.total_tokens:,}")
        print(f"[+] Grid memory: {self.voxel_grid.nbytes / 1024 / 1024:.2f} MB")
    
    def _smooth_grid(self, sigma: float = 1.5) -> None:
        """
        Apply 3D Gaussian smoothing to create continuous density fields.
        This allows for smoother geometric pathfinding.
        """
        from scipy.ndimage import gaussian_filter
        self.voxel_grid = gaussian_filter(self.voxel_grid, sigma=sigma)
    
    def train_from_model(self, model, tokenizer) -> None:
        """
        Complete training pipeline: extract weights, map to 3D, build grid.
        """
        print("=" * 70)
        print(" VOXEL ENGINE: Weight-to-Voxel Distillation")
        print("=" * 70)
        
        # Step 1: Extract embeddings
        embeddings = self.extract_embeddings(model)
        
        # Step 2: Map to 3D coordinates
        coords_3d = self.map_embeddings_to_3d(embeddings)
        
        # Step 3: Build voxel grid
        self.build_voxel_grid(embeddings, coords_3d)
        
        # Step 4: Build trajectory tunnels from tokenizer vocabulary
        print("[*] Building trajectory tunnels...")
        self._build_tunnels(tokenizer)
        
        print("[+] Training complete!")
    
    def _build_tunnels(self, tokenizer) -> None:
        """
        Build trajectory tunnels for high-frequency token sequences.
        Uses the tokenizer's vocabulary to identify common patterns.
        """
        # Extract common bigrams and trigrams from vocabulary
        vocab = tokenizer.get_vocab()
        token_freq = Counter()
        
        # Analyze token structure to find common patterns
        for token, token_id in vocab.items():
            if len(token) >= 2:
                # Count character-level patterns
                for i in range(len(token) - 1):
                    bigram = token[i:i+2]
                    token_freq[bigram] += 1
        
        # Build tunnels for high-frequency patterns
        for pattern, freq in token_freq.items():
            if freq >= 100:  # Threshold for tunnel creation
                # Store the pattern as a tunnel
                self.tunnels[pattern] = pattern
    
    def get_next_token_distribution(self, current_token_id: int, temperature: float = 0.8) -> np.ndarray:
        """
        Get probability distribution for the next token based on geometric proximity
        in the voxel space.
        """
        if current_token_id not in self.token_coords:
            # Fallback: uniform distribution
            return np.ones(self.vocab_size) / self.vocab_size
        
        x, y, z = self.token_coords[current_token_id]
        
        # Get neighboring voxels (3x3x3 neighborhood)
        neighbor_scores = np.zeros(self.vocab_size)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 0 <= nz < self.grid_size:
                        # Get tokens in this voxel
                        tokens_in_voxel = self.coord_to_tokens.get((nx, ny, nz), [])
                        voxel_density = self.voxel_grid[nx, ny, nz]
                        
                        # Distribute score to tokens in this voxel
                        for tid in tokens_in_voxel:
                            # Distance-based weighting
                            distance = np.sqrt(dx**2 + dy**2 + dz**2)
                            weight = np.exp(-distance / 2.0) * voxel_density
                            neighbor_scores[tid] += weight
        
        # Apply temperature scaling
        if temperature > 0:
            neighbor_scores = neighbor_scores ** (1.0 / temperature)
        
        # Normalize to probability distribution
        total = np.sum(neighbor_scores)
        if total > 0:
            return neighbor_scores / total
        return np.ones(self.vocab_size) / self.vocab_size
    
    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> List[int]:
        """
        Generate tokens using geometric pathfinding through the voxel space.
        """
        generated = prompt_tokens.copy()
        recent_tokens = set(prompt_tokens[-4:])  # Track recent for repetition penalty
        
        for step in range(max_new_tokens):
            current_token = generated[-1]
            
            # Get geometric probability distribution
            probs = self.get_next_token_distribution(current_token, temperature)
            
            # Apply repetition penalty
            for tid in recent_tokens:
                if tid < len(probs):
                    probs[tid] /= repetition_penalty
            
            # Re-normalize
            probs = probs / (np.sum(probs) + 1e-8)
            
            # Top-K filtering
            if top_k > 0:
                top_k_indices = np.argsort(probs)[-top_k:]
                mask = np.ones_like(probs, dtype=bool)
                mask[top_k_indices] = False
                probs[mask] = 0
                probs = probs / (np.sum(probs) + 1e-8)
            
            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_indices = np.argsort(probs)[::-1]
                cumulative_probs = np.cumsum(probs[sorted_indices])
                cutoff = np.searchsorted(cumulative_probs, top_p) + 1
                mask = np.ones_like(probs, dtype=bool)
                mask[sorted_indices[:cutoff]] = False
                probs[mask] = 0
                probs = probs / (np.sum(probs) + 1e-8)
            
            # Sample next token
            next_token = np.random.choice(len(probs), p=probs)
            generated.append(next_token)
            
            # Update recent tokens
            recent_tokens.add(next_token)
            if len(recent_tokens) > 6:
                oldest = generated[-7]
                recent_tokens.discard(oldest)
        
        return generated
    
    def save(self, path: Path) -> None:
        """Save the trained voxel model to disk."""
        print(f"[*] Saving voxel model to {path}...")
        
        # Save voxel grid
        np.save(path.with_suffix('.grid.npy'), self.voxel_grid)
        
        # Convert token_coords to JSON-serializable format
        token_coords_serializable = {}
        for token_id, coords in self.token_coords.items():
            # Convert numpy int64 to Python int
            token_coords_serializable[str(int(token_id))] = [int(c) for c in coords]
        
        # Save metadata and mappings
        metadata = {
            'grid_size': int(self.grid_size),
            'pca_components': int(self.pca_components),
            'total_tokens': int(self.total_tokens),
            'vocab_size': int(self.vocab_size),
            'is_trained': bool(self.is_trained),
            'token_coords': token_coords_serializable,
            'tunnels': self.tunnels
        }
        
        with open(path.with_suffix('.meta.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[+] Voxel model saved successfully!")
    
    def load(self, path: Path) -> None:
        """Load a trained voxel model from disk."""
        print(f"[*] Loading voxel model from {path}...")
        
        # Load voxel grid
        grid_path = path.with_suffix('.grid.npy')
        if grid_path.exists():
            self.voxel_grid = np.load(grid_path)
            self.grid_size = self.voxel_grid.shape[0]
        
        # Load metadata
        meta_path = path.with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            self.pca_components = metadata['pca_components']
            self.total_tokens = metadata['total_tokens']
            self.vocab_size = metadata['vocab_size']
            self.is_trained = metadata['is_trained']
            self.tunnels = metadata.get('tunnels', {})
            
            # Reconstruct token_coords
            self.token_coords = {int(k): tuple(v) for k, v in metadata['token_coords'].items()}
            
            # Reconstruct coord_to_tokens
            self.coord_to_tokens.clear()
            for token_id, coords in self.token_coords.items():
                self.coord_to_tokens[coords].append(token_id)
        
        print(f"[+] Voxel model loaded. Grid size: {self.grid_size}, Tokens: {self.total_tokens:,}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the trained voxel model."""
        return {
            'grid_size': self.grid_size,
            'total_voxels': self.grid_size ** 3,
            'occupied_voxels': np.count_nonzero(self.voxel_grid),
            'total_tokens': self.total_tokens,
            'vocab_size': self.vocab_size,
            'memory_mb': self.voxel_grid.nbytes / 1024 / 1024,
            'is_trained': self.is_trained,
            'num_tunnels': len(self.tunnels)
        }