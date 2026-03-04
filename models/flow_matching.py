import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """將時間 t (0~1) 轉換成向量，讓 MLP 知道現在走到哪了"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        # 關閉原生的 affine 參數，交給時間 t 來控制
        self.norm = nn.LayerNorm(dim, elementwise_affine=False) 
        # 新增一個小型 MLP，把 t_emb 轉成 scale 和 shift
        self.ada_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2)
        )

    def forward(self, x, t_emb):
        # 將 t_emb 投影並切兩半
        scale, shift = self.ada_mlp(t_emb).chunk(2, dim=-1)
        # AdaLN 的核心公式
        h = self.norm(x) * (1 + scale) + shift 
        return x + self.proj(h)

class VectorFieldNetwork(nn.Module):
    """
    輸入: x (Noisy Feature), t (Time)
    輸出: v (Velocity / Direction)
    """
    def __init__(self, in_dim=1024, hidden_dim=1024, num_layers=6):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            Block(hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x, t):
        # 1. 處理時間 Embedding
        t_emb = self.time_mlp(t)
        
        # 2. 處理輸入特徵
        h = self.input_proj(x)
        
        # 3. 融合時間與特徵 + 通過 ResNet Blocks
        for block in self.blocks:
            h = block(h, t_emb)
            
        return self.output_proj(h)

# ==========================================
# 🚀 採樣器 (Euler Solver)
# ==========================================
@torch.no_grad()
def sample_flow_matching(model, z, steps=10):
    model.eval()
    x = z.clone()
    # 建立 0.0 到 1.0 的精準時間點
    times = torch.linspace(0.0, 1.0, steps + 1, device=z.device)
    
    for i in range(steps):
        t_curr = times[i]
        t_next = times[i+1]
        dt = t_next - t_curr
        
        t = torch.full((z.shape[0],), t_curr, device=z.device)
        v = model(x, t)
        
        # Euler 積分
        x = x + v * dt 
    return x