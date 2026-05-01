import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
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

def modulate(x, shift, scale):
    # 將 shift 和 scale 擴展至序列維度: [B, 1, D]
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """SOTA: Transformer Block with AdaLN-Zero"""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        # AdaLN 產生 6 個控制參數
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-Attention 加上時間調變
        x_mod1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_mod1, x_mod1, x_mod1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # FFN 加上時間調變
        x_mod2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_mod2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

class VectorFieldDiT(nn.Module):
    """
    SOTA Flow Matching 模型
    輸入: x (Noisy Sequence: [B, S, D]), t (Time)
    輸出: v (Velocity Sequence: [B, S, D])
    """
    def __init__(self, seq_len=16, in_dim=512, hidden_dim=1024, num_layers=8, num_heads=16):
        super().__init__()
        # 1. 特徵投影 & 位置編碼
        self.x_embedder = nn.Linear(in_dim, hidden_dim)
        # 🌟 讓模型知道每個 token 代表哪個空間位置
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim)) 
        
        # 2. 時間編碼
        self.t_embedder = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 3. DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        # 4. 輸出層 (AdaLN-Zero)
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
            nn.Linear(hidden_dim, in_dim)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, x, t):
        # x shape: [B, seq_len, in_dim]
        c = self.t_embedder(t) # [B, hidden_dim]
        
        # 加上位置編碼
        x = self.x_embedder(x) + self.pos_embed 
        
        for block in self.blocks:
            x = block(x, c)
            
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.final_layer[0](x), shift, scale)
        x = self.final_layer[1](x)
        return x

# ==========================================
# 🚀 採樣器 (不變，但支援 3D 張量)
# ==========================================
@torch.no_grad()
def sample_flow_matching(model, z, steps=10):
    model.eval()
    x = z.clone() # z 的形狀是 [B, seq_len, in_dim]
    times = torch.linspace(0.0, 1.0, steps + 1, device=z.device)
    
    for i in range(steps):
        t_curr = times[i]
        t_next = times[i+1]
        dt = t_next - t_curr
        
        t = torch.full((z.shape[0],), t_curr, device=z.device)
        v = model(x, t) # 預測速度場
        x = x + v * dt 
    return x