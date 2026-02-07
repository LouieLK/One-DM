import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange
from mamba_ssm import Mamba
from models.resnet_dilation import resnet18 as resnet18_dilation
from models.transformer import PositionalEncoding, PositionalEncoding2D

# --------------------------------------------------------------------------------
# Mamba Components (Adapters)
# --------------------------------------------------------------------------------

class BidirectionalMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.output_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (Batch, SeqLen, Dim)
        residual = x
        x = self.norm(x)
        out_fwd = self.forward_mamba(x)
        x_rev = torch.flip(x, dims=[1])
        out_bwd = self.backward_mamba(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[1])
        out = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.output_proj(out)
        return out + residual

class MambaEncoder(nn.Module):
    def __init__(self, d_model, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(d_model) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        # x: (Batch, SeqLen, Dim)
        for layer in self.layers:
            x = layer(x)
        return x

class MambaDecoder(nn.Module):
    def __init__(self, d_model, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(d_model) for _ in range(num_layers)
        ])
    
    def forward(self, tgt, memory):
        # tgt: Content (Batch, L_c, Dim)
        # memory: Style (Batch, L_s, Dim)
        # Concatenate: [Style, Content] as Context
        x = torch.cat([memory, tgt], dim=1)
        for layer in self.layers:
            x = layer(x)
        # Return only the content part (last L_c tokens)
        return x[:, memory.size(1):, :]

# --------------------------------------------------------------------------------
# Main Class: MambaStyleFusion (Aligned with Mix_TR)
# --------------------------------------------------------------------------------

class MambaStyleFusion(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=2048, dropout=0.1, activation="relu", return_intermediate_dec=False,
                 normalize_before=True):
        super(MambaStyleFusion, self).__init__()
        
        self.d_model = d_model
        
        # --- Mamba Encoders (Replacing TransformerEncoder) ---
        self.style_encoder = MambaEncoder(d_model, num_encoder_layers)
        self.fre_encoder = MambaEncoder(d_model, num_encoder_layers)

        # --- Mamba Decoders (Replacing TransformerDecoder) ---
        self.decoder = MambaDecoder(d_model, num_decoder_layers)
        self.fre_decoder = MambaDecoder(d_model, num_decoder_layers)
        
        self.add_position1D = PositionalEncoding(dropout=0.1, dim=d_model) 
        self.add_position2D = PositionalEncoding2D(dropout=0.1, d_model=d_model) 
        
        # MLPs (Aligned with Mix_TR)
        # Note: Using d_model instead of hardcoded 512 to ensure compatibility if d_model changes
        self.high_pro_mlp = nn.Sequential(
            nn.Linear(d_model, 4096), nn.GELU(), nn.Linear(4096, 256))
        self.low_pro_mlp = nn.Sequential(
            nn.Linear(d_model, 4096), nn.GELU(), nn.Linear(4096, 256))
        self.low_feature_filter = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

        self._reset_parameters()

        # --- Feature Encoders (ResNet18) ---
        self.Feat_Encoder = self.initialize_resnet18()
        self.style_dilation_layer = resnet18_dilation().conv5_x
        
        self.freq_encoder = self.initialize_resnet18()
        self.freq_dilation_layer = resnet18_dilation().conv5_x

        self.content_encoder = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] + list(models.resnet18(weights='ResNet18_Weights.DEFAULT').children())[1:-2]))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def initialize_resnet18(self):
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.layer4 = nn.Identity()
        resnet.fc = nn.Identity()
        resnet.avgpool = nn.Identity()
        return resnet

    def process_style_feature(self, encoder, dilation_layer, style, add_position2D, style_encoder):
        style = encoder(style)
        # ResNet layer3 output has 256 channels.
        # Mix_TR rearranges with hardcoded c=256. We ensure d_model matches or this works.
        # rearrange 'n (c h w) -> n c h w' is effectively identity if input is already NCHW
        # We explicitly set shape to (Batch, Seq, Dim) for Mamba
        
        # style = rearrange(style, 'n (c h w) ->n c h w', c=256, h=4).contiguous() # From Mix_TR, kept if needed
        style = dilation_layer(style)
        style = add_position2D(style) # Returns (N, C, H, W)
        
        # Reshape for Mamba: (Batch, Sequence, Dim)
        style = rearrange(style, 'n c h w -> n (h w) c').contiguous()
        style = style_encoder(style)
        return style

    def get_low_style_feature(self, style):
        return self.process_style_feature(self.Feat_Encoder, self.style_dilation_layer, style, self.add_position2D, self.style_encoder)

    def get_high_style_feature(self, laplace):
        return self.process_style_feature(self.freq_encoder, self.freq_dilation_layer, laplace, self.add_position2D, self.fre_encoder)

    def forward(self, style, laplace, content):
        # --- High Frequency Processing ---
        anchor_style = style[:, 0, :, :].clone().unsqueeze(1).contiguous()
        anchor_high = laplace[:, 0, :, :].clone().unsqueeze(1).contiguous()
        
        anchor_high_feature = self.get_high_style_feature(anchor_high) # (B, L, D)

        anchor_high_nce = self.high_pro_mlp(anchor_high_feature)
        anchor_high_nce = torch.mean(anchor_high_nce, dim=1) # Mean over sequence

        pos_style = style[:, 1, :, :].clone().unsqueeze(1).contiguous()
        pos_high = laplace[:, 1, :, :].clone().unsqueeze(1).contiguous()
        pos_high_feature = self.get_high_style_feature(pos_high)

        pos_high_nce = self.high_pro_mlp(pos_high_feature)
        pos_high_nce = torch.mean(pos_high_nce, dim=1)
  
        high_nce_emb = torch.stack([anchor_high_nce, pos_high_nce], dim=1)
        high_nce_emb = nn.functional.normalize(high_nce_emb, p=2, dim=2)

        # --- Low Frequency Processing ---
        anchor_low = anchor_style
        anchor_low_feature = self.get_low_style_feature(anchor_low)
        anchor_mask = self.low_feature_filter(anchor_low_feature)
        anchor_low_feature = anchor_low_feature * anchor_mask
        
        anchor_low_nce = self.low_pro_mlp(anchor_low_feature)
        anchor_low_nce = torch.mean(anchor_low_nce, dim=1)

        pos_low = pos_style 
        pos_low_feature = self.get_low_style_feature(pos_low)
        pos_mask = self.low_feature_filter(pos_low_feature)
        pos_low_feature = pos_low_feature * pos_mask
        
        pos_low_nce = self.low_pro_mlp(pos_low_feature)
        pos_low_nce = torch.mean(pos_low_nce, dim=1)

        low_nce_emb = torch.stack([anchor_low_nce, pos_low_nce], dim=1)
        low_nce_emb = nn.functional.normalize(low_nce_emb, p=2, dim=2)

        # --- Content Encoding ---
        content = rearrange(content, 'n t h w ->(n t) 1 h w').contiguous()
        content = self.content_encoder(content)
        # Reshape to (Batch, Seq, Dim) for Mamba
        content = rearrange(content, '(n t) c h w -> n t (c h w)', n=style.shape[0]).contiguous()
        content = self.add_position1D(content)
        
        # --- Fusion ---
        style_hs = self.decoder(content, anchor_low_feature) # (B, L, D)
        hs = self.fre_decoder(style_hs, anchor_high_feature) # (B, L, D)
        
        # Mix_TR returns (Batch, Seq, Dim) implicitly via permute(1,0,2) on Transformer output (Seq, Batch, Dim)
        # Mamba output is already (Batch, Seq, Dim), so no permutation needed
        return hs, high_nce_emb, low_nce_emb

    def generate(self, style, laplace, content):
        if style.shape[1] == 1:
            anchor_style = style
            anchor_high = laplace
        else:
            anchor_style = style[:, 0, :, :].unsqueeze(1).contiguous()
            anchor_high = laplace[:, 0, :, :].unsqueeze(1).contiguous()
        
        # Get Features
        anchor_high_feature = self.get_high_style_feature(anchor_high)
        
        anchor_low = anchor_style
        anchor_low_feature = self.get_low_style_feature(anchor_low)
        anchor_mask = self.low_feature_filter(anchor_low_feature)
        anchor_low_feature = anchor_low_feature * anchor_mask

        # Content Encoder
        content = rearrange(content, 'n t h w ->(n t) 1 h w').contiguous()
        content = self.content_encoder(content)
        content = rearrange(content, '(n t) c h w -> n t (c h w)', n=style.shape[0]).contiguous()
        content = self.add_position1D(content)
        
        # Fusion
        style_hs = self.decoder(content, anchor_low_feature)
        hs = self.fre_decoder(style_hs, anchor_high_feature)
        
        return hs