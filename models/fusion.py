import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models
from models.transformer import *
from einops import rearrange, repeat
import math
from models.resnet_dilation import resnet18 as resnet18_dilation

### merge the handwriting style and printed content
class Mix_TR(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=2048, dropout=0.1, activation="relu", return_intermediate_dec=False,
                 normalize_before=True):
        super(Mix_TR, self).__init__()
        
        self.d_model = d_model # 保存 d_model 以供使用
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        style_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.style_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, style_norm)

        fre_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.fre_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, fre_norm)

        ### fusion the content and style in the transformer decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec)
        
        fre_decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.fre_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, fre_decoder_norm,
                                        return_intermediate=return_intermediate_dec)
        
        self.add_position1D = PositionalEncoding(dropout=0.1, dim=d_model) # add 1D position encoding
        self.add_position2D = PositionalEncoding2D(dropout=0.1, d_model=d_model) # add 2D position encoding
        
        # 注意: 這裡的 512 可能是硬編碼，建議確認是否需改為 d_model
        # 如果您的 d_model 是 256 但 ResNet 輸出是 512，這裡不用動
        self.high_pro_mlp = nn.Sequential(
            nn.Linear(self.d_model, 4096), nn.GELU(), nn.Linear(4096, 256))
        self.low_pro_mlp = nn.Sequential(
            nn.Linear(self.d_model, 4096), nn.GELU(), nn.Linear(4096, 256))
        self.low_feature_filter = nn.Sequential(nn.Linear(self.d_model, 1), nn.Sigmoid())

        # [修改] 為了支援任意解析度，Null Feature 統一改為長度 1，後續再動態擴展
        self.null_low_feature = nn.Parameter(torch.randn(1, 1, d_model))
        self.null_high_feature = nn.Parameter(torch.randn(1, 1, d_model))
        self.null_content_feature = nn.Parameter(torch.randn(1, 1, d_model))

        self._reset_parameters()

        ### low frequency style encoder
        self.Feat_Encoder = self.initialize_resnet18()
        self.style_dilation_layer = resnet18_dilation().conv5_x
        # 🌟 [新增] 將 ResNet 輸出的 512 維投影到您設定的 EMB_DIM
        self.style_proj = nn.Conv2d(512, self.d_model, kernel_size=1)

        ### hig frequency style encoder
        self.freq_encoder = self.initialize_resnet18()
        self.freq_dilation_layer = resnet18_dilation().conv5_x
        # 🌟 [新增] 將 ResNet 輸出的 512 維投影到您設定的 EMB_DIM
        self.freq_proj = nn.Conv2d(512, self.d_model, kernel_size=1)

        ### content encoder
        # self.content_encoder = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(models.resnet18(weights='ResNet18_Weights.DEFAULT').children())[1:-2]))
        self.content_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(models.resnet18(weights='ResNet18_Weights.DEFAULT').children())[1:-2],
            # [新增] 無論輸入的字型圖片多大，都強制提取全局結構特徵為 1x1
            nn.AdaptiveAvgPool2d((1, 1)), 
            # [新增] 完美將 ResNet 的 256 通道映射到您在 YAML 設定的 EMB_DIM (d_model)
            nn.Conv2d(256, self.d_model, kernel_size=1) 
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def initialize_resnet18(self,):
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.layer4 = nn.Identity()
        resnet.fc = nn.Identity()
        resnet.avgpool = nn.Identity()
        return resnet

    # [修改] 函數簽名增加 proj_layer
    def process_style_feature(self, encoder, dilation_layer, proj_layer, style, add_position2D, style_encoder):
        style = encoder(style)
        
        # 🌟 [動態計算] 自動計算特徵圖的長寬，相容 64x64 (h=4) 與 128x128 (h=8)
        spatial_dim = int(math.sqrt(style.shape[1] // 256))
        style = rearrange(style, 'n (c h w) ->n c h w', c=256, h=spatial_dim).contiguous()
        
        style = dilation_layer(style)
        style = proj_layer(style) # 🌟 [新增] 512 維降至 EMB_DIM
        
        style = add_position2D(style)
        style = rearrange(style, 'n c h w ->(h w) n c').contiguous()
        style = style_encoder(style)
        return style


    def get_low_style_feature(self, style):
        return self.process_style_feature(self.Feat_Encoder, self.style_dilation_layer, self.style_proj, style, self.add_position2D, self.style_encoder)

    def get_high_style_feature(self, laplace):
        return self.process_style_feature(self.freq_encoder, self.freq_dilation_layer, self.freq_proj, laplace, self.add_position2D, self.fre_encoder)

    def get_style_vectors(self, style, laplace):
        if style.shape[1] == 1:
            anchor_style = style
            anchor_high = laplace
        else:
            anchor_style = style[:, 0, :, :].unsqueeze(1).contiguous()
            anchor_high = laplace[:, 0, :, :].unsqueeze(1).contiguous()

        anchor_high_feature = self.get_high_style_feature(anchor_high) 
        high_vec = torch.mean(anchor_high_feature, dim=0) 

        anchor_low = anchor_style
        anchor_low_feature = self.get_low_style_feature(anchor_low)
        anchor_mask = self.low_feature_filter(anchor_low_feature)
        anchor_low_feature = anchor_low_feature * anchor_mask 
        low_vec = torch.mean(anchor_low_feature, dim=0) 
        
        return low_vec, high_vec

    
    def forward(self, style, laplace, content):
        # 檢查是否為 Unconditional (Trainer 傳入全零圖片)
        # 判斷標準：style 的絕對值總和是否接近 0
        is_style_uncond = (torch.sum(torch.abs(style)) < 1e-6)
        batch_size = style.shape[0]

        if is_style_uncond:
            # === CFG Unconditional Path ===
            # 使用 Learnable Null Embedding 擴展到 batch size
            # shape: (16, B, d_model)
            # 🌟 [動態計算序列長度]
            is_vector_input = (style.dim() == 2) if hasattr(style, 'dim') else False
            seq_len = 1 if is_vector_input else (style.shape[2] // 16) * (style.shape[3] // 16)
            
            anchor_high_feature = self.null_high_feature.expand(seq_len, batch_size, -1)
            anchor_low_feature = self.null_low_feature.expand(seq_len, batch_size, -1)
            
            # 對於 NCE Loss 的 embedding，無條件時 Loss 不計算，給 dummy 即可
            dummy_nce = torch.zeros(batch_size, 256, device=style.device) # 假設 MLP 輸出 256
            high_nce_emb = torch.stack([dummy_nce, dummy_nce], dim=1)
            low_nce_emb = torch.stack([dummy_nce, dummy_nce], dim=1)
            
        else:
            # === Normal Conditional Path ===
            # get the high frequency and style feature
            anchor_style = style[:, 0, :, :].clone().unsqueeze(1).contiguous()
            anchor_high = laplace[:, 0, :, :].clone().unsqueeze(1).contiguous()
            anchor_high_feature = self.get_high_style_feature(anchor_high) # t n c

            anchor_high_nce = self.high_pro_mlp(anchor_high_feature) # t n c
            anchor_high_nce = torch.mean(anchor_high_nce, dim=0) # n c

            pos_style = style[:, 1, :, :].clone().unsqueeze(1).contiguous()
            pos_high = laplace[:, 1, :, :].clone().unsqueeze(1).contiguous()
            pos_high_feature = self.get_high_style_feature(pos_high) # t n c

            pos_high_nce = self.high_pro_mlp(pos_high_feature) # t n c
            pos_high_nce = torch.mean(pos_high_nce, dim=0) # n c
    
            high_nce_emb = torch.stack([anchor_high_nce, pos_high_nce], dim=1) # B 2 C
            high_nce_emb = nn.functional.normalize(high_nce_emb, p=2, dim=2)

            # get the low frequency and style feature
            anchor_low = anchor_style
            anchor_low_feature = self.get_low_style_feature(anchor_low)
            anchor_mask = self.low_feature_filter(anchor_low_feature)
            anchor_low_feature = anchor_low_feature * anchor_mask
            anchor_low_nce = self.low_pro_mlp(anchor_low_feature) # t n c
            anchor_low_nce = torch.mean(anchor_low_nce, dim=0)

            pos_low = pos_style 
            pos_low_feature = self.get_low_style_feature(pos_low)
            pos_mask = self.low_feature_filter(pos_low_feature)
            pos_low_feature = pos_low_feature * pos_mask
            pos_low_nce = self.low_pro_mlp(pos_low_feature)
            pos_low_nce = torch.mean(pos_low_nce, dim=0)

            low_nce_emb = torch.stack([anchor_low_nce, pos_low_nce], dim=1) # B 2 C
            low_nce_emb = nn.functional.normalize(low_nce_emb, p=2, dim=2)

        # =========== [修改後] 加入 Unconditional 判斷 ===========
        is_content_uncond = (torch.sum(torch.abs(content)) < 1e-6)
        t_len = content.shape[1] # 取得序列長度 (也就是 max_len，通常是 1)
        
        if is_content_uncond:
            # CFG Content Unconditional Path
            # 直接使用空殼向量並展開至對應的 sequence_length 與 batch_size
            content_feat = self.null_content_feature.expand(t_len, batch_size, -1)
        else:
            # 正常處理路徑
            content = rearrange(content, 'n t h w ->(n t) 1 h w').contiguous()
            content = self.content_encoder(content)
            content = rearrange(content, '(n t) c h w ->t n (c h w)', n=batch_size).contiguous()
            content_feat = self.add_position1D(content)
        
        # 把原本傳入 decoder 的 `content` 替換成 `content_feat`
        style_hs = self.decoder(content_feat, anchor_low_feature, tgt_mask=None)
        hs = self.fre_decoder(style_hs[0], anchor_high_feature, tgt_mask=None)
        
        return hs[0].permute(1, 0, 2).contiguous(), high_nce_emb, low_nce_emb # n t c
    
    def generate(self, style, laplace, content):
        is_vector_input = (style.dim() == 2)
        # 檢查是否為 Unconditional (Inference 時傳入全零)
        is_style_uncond = (torch.sum(torch.abs(style)) < 1e-6)
        batch_size = style.shape[0]

        if is_style_uncond:
             # === CFG Unconditional Path ===
             # 擴展 Null Embedding
             # 🌟 [動態計算序列長度]
            is_vector_input = (style.dim() == 2) if hasattr(style, 'dim') else False
            seq_len = 1 if is_vector_input else (style.shape[2] // 16) * (style.shape[3] // 16)
            
            anchor_high_feature = self.null_high_feature.expand(seq_len, batch_size, -1)
            anchor_low_feature = self.null_low_feature.expand(seq_len, batch_size, -1)
             
        elif is_vector_input:
            # === Mode 2: Vector Input (from Flow) ===
            anchor_low_feature = style.unsqueeze(0)  # [1, N, 512]
            anchor_high_feature = laplace.unsqueeze(0) # [1, N, 512]
            
        else:
            # === Mode 1: Image Input (Original) ===
            if style.shape[1] == 1:
                anchor_style = style
                anchor_high = laplace
            else:
                anchor_style = style[:, 0, :, :].unsqueeze(1).contiguous()
                anchor_high = laplace[:, 0, :, :].unsqueeze(1).contiguous()
            
            # get the high frequency style feature
            anchor_high_feature = self.get_high_style_feature(anchor_high) # t n c
            
            # get the low frequency style feature
            anchor_low = anchor_style
            anchor_low_feature = self.get_low_style_feature(anchor_low)
            anchor_mask = self.low_feature_filter(anchor_low_feature)
            anchor_low_feature = anchor_low_feature * anchor_mask

        # =========== [修改後] 加入 Unconditional 判斷 ===========
        is_content_uncond = (torch.sum(torch.abs(content)) < 1e-6)
        t_len = content.shape[1] # 取得序列長度 (也就是 max_len，通常是 1)
        
        if is_content_uncond:
            # CFG Content Unconditional Path
            # 直接使用空殼向量並展開至對應的 sequence_length 與 batch_size
            content_feat = self.null_content_feature.expand(t_len, batch_size, -1)
        else:
            # 正常處理路徑
            content = rearrange(content, 'n t h w ->(n t) 1 h w').contiguous()
            content = self.content_encoder(content)
            content = rearrange(content, '(n t) c h w ->t n (c h w)', n=batch_size).contiguous()
            content_feat = self.add_position1D(content)
        
        # 把原本傳入 decoder 的 `content` 替換成 `content_feat`
        style_hs = self.decoder(content_feat, anchor_low_feature, tgt_mask=None)
        hs = self.fre_decoder(style_hs[0], anchor_high_feature, tgt_mask=None)
        
        return hs[0].permute(1, 0, 2).contiguous()