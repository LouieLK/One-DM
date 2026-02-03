"""
程式名稱: models/mamba_style.py
功能描述: 
    修正後的 Vision Mamba 架構，適配 One-DM 的 UNet 接口。
    現在 forward() 可以直接接收 (style, laplace, content) 並回傳 (context, high, low)。
"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class PatchEmbed(nn.Module):
    """將 2D 圖片切塊並展平為 1D 序列"""
    def __init__(self, img_size=128, patch_size=4, in_chans=2, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        # 使用 Conv2d 做 Patch Projection
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, D, H/P, W/P] -> [B, D, N] -> [B, N, D]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VimBlock(nn.Module):
    """Vision Mamba Block (包含雙向掃描)"""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # 正向 Mamba
        self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        # 反向 Mamba
        self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, x):
        identity = x
        x = self.norm(x)

        # 1. 正向掃描
        out_fwd = self.mamba_fwd(x)
        
        # 2. 反向掃描
        x_flip = torch.flip(x, dims=[1])
        out_bwd = self.mamba_bwd(x_flip)
        out_bwd = torch.flip(out_bwd, dims=[1])
        
        # 3. 融合
        out = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.proj(out)
        
        return out + identity

class MambaStyleEncoder(nn.Module):
    def __init__(self, 
                 img_size=64,       # 注意：請確認這裡跟您的資料集圖片大小一致
                 in_chans=2,        # 預設接收 Style + Laplace
                 embed_dim=512,     # 內部維度
                 depth=8,           # Mamba 層數
                 out_dim=512):      # 輸出向量維度
        super().__init__()
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim
        )
        
        # Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        
        # 2. Mamba Encoder Stack
        self.blocks = nn.ModuleList([
            VimBlock(dim=embed_dim, d_state=16, expand=2) 
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 3. Output Heads
        self.head_low = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, out_dim)
        )
        
        self.head_high = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, out_dim)
        )

        # 初始化權重
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, style, laplace, content=None):
        """
        修正後的 forward，符合 unet.py 的呼叫方式：
        context, high, low = self.mix_net(style, laplace, content)
        """
        
        # 1. 處理輸入圖片
        # 根據 in_chans 決定是否要加入 content (通常 Style Encoder 不需要 content)
        if self.in_chans == 3 and content is not None:
            x_img = torch.cat([style, laplace, content], dim=1)
        else:
            x_img = torch.cat([style, laplace], dim=1)
            
        # 2. Patch Embedding
        x = self.patch_embed(x_img) # [B, N, D]
        
        # 3. 加入 CLS Token 與 Positional Embedding
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # [B, N+1, D]
        
        # 簡單的 Pos Embed 插值 (防止輸入尺寸變化導致報錯)
        if x.shape[1] != self.pos_embed.shape[1]:
            # 如果序列長度不對，這裡可以加插值邏輯，暫時先直接加
            # 在固定尺寸訓練下通常不會發生
            x = x + self.pos_embed[:, :x.shape[1], :]
        else:
            x = x + self.pos_embed
        
        # 4. Mamba Layers Forward
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # 5. 分離特徵
        cls_out = x[:, 0]        # [B, D] -> 用於 High Level
        patch_out = x[:, 1:]     # [B, N, D] -> 用於 Context
        
        # 6. 生成輸出向量
        high_vec = self.head_high(cls_out)           # High-level Style (全域)
        low_vec = self.head_low(patch_out.mean(dim=1)) # Low-level Style (筆觸，取平均)
        
        # 回傳三個值：Context (序列), High vector, Low vector
        return patch_out, high_vec, low_vec