"""
程式名稱: train_flow_ViT_online.py
功能描述: 
    同步 HandwritingDataset 邏輯，即時透過 U-Net 萃取特徵訓練 DiT。
    支援字元反轉對齊、多尺度 Content 獲取，並包含完整的訓練與驗證機制。
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

# 引用專案模組
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from models.unet import UNetModel
from data_loader.loader import HandwritingDataset, ContentData

# 引入 DiT 模型
from models.DiT import VectorFieldDiT, sample_flow_matching

def setup_logger(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = os.path.join(save_dir, "train_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def main(opt):
    device = torch.device(opt.device)
    logger = setup_logger(opt.save_dir)
    
    # 載入設定檔
    cfg_from_file(opt.cfg)
    assert_and_infer_cfg()
    
    logger.info("🚀 啟動 DiT Online 訓練模式 (對齊 HandwritingDataset 邏輯)")

    # ==========================================================
    # 1. 載入並凍結 U-Net (特徵引擎)
    # ==========================================================
    unet = UNetModel(
        in_channels=cfg.MODEL.IN_CHANNELS, 
        model_channels=cfg.MODEL.EMB_DIM, 
        out_channels=cfg.MODEL.OUT_CHANNELS, 
        num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
        attention_resolutions=cfg.MODEL.ATTENTION_RESOLUTIONS, 
        channel_mult=cfg.MODEL.CHANNEL_MULT, 
        num_heads=cfg.MODEL.NUM_HEADS, 
        context_dim=cfg.MODEL.EMB_DIM,
    ).to(device)

    ckpt = torch.load(opt.unet_ckpt, map_location='cpu')
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    unet.load_state_dict(new_state_dict, strict=False)
    unet.eval().requires_grad_(False)
    
    # ==========================================================
    # 2. 準備 Dataset (同步 Loader 邏輯)
    # ==========================================================
    HandwritingDataset.set_global_config(cfg)
    
    content_data_engine = ContentData(split='train', content_type='kaifont', cfg=cfg)
    
    train_dataset = HandwritingDataset(split='train', use_latent=True) 
    val_dataset = HandwritingDataset(split='test', use_latent=True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True, 
        collate_fn=train_dataset.collate_fn_ 
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=val_dataset.collate_fn_
    )

    # ==========================================================
    # 3. 初始化 DiT 模型
    # ==========================================================
    with torch.no_grad():
        dummy_batch = next(iter(train_loader))
        dummy_feat = unet.mix_net.get_style_vectors(dummy_batch['style'].to(device), dummy_batch['laplace'].to(device))
        seq_len, flow_dim = dummy_feat.shape[1], dummy_feat.shape[2]

    vf_net = VectorFieldDiT(seq_len=seq_len, in_dim=flow_dim, hidden_dim=1024, num_layers=8, num_heads=16).to(device)
    optimizer = optim.AdamW(vf_net.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    best_val_loss = float('inf')

    for epoch in range(opt.epochs):
        # ------------------------------------------------------
        # [訓練階段 Train]
        # ------------------------------------------------------
        vf_net.train()
        epoch_loss = 0.0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epochs} [Train]")
        
        for data in pbar:
            style_img = data['style'].to(device)
            laplace_img = data['laplace'].to(device)
            full_content_tensor = data['content'].to(device) 
            content_img = full_content_tensor[:, 0:1, :, :] 
            
            with torch.no_grad():
                x1 = unet.mix_net.get_style_vectors(style_img, laplace_img).detach()
            
            b = x1.shape[0]
            x0 = torch.randn_like(x1)
            t = torch.rand(b, device=device)
            t_exp = t.view(b, 1, 1)
            x_t = (1 - t_exp) * x0 + t_exp * x1
            target_v = x1 - x0
            
            pred_v = vf_net(x_t, t, content_img=content_img)
            loss = torch.mean((pred_v - target_v) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vf_net.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            train_steps += 1
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        avg_train_loss = epoch_loss / train_steps

        # ------------------------------------------------------
        # [驗證階段 Validation]
        # ------------------------------------------------------
        vf_net.eval()
        val_loss = 0.0
        val_steps = 0
        
        # 關閉梯度計算，節省顯存並加速
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{opt.epochs} [Val]", leave=False):
                # 取得 Validation 數據
                style_img = data['style'].to(device)
                laplace_img = data['laplace'].to(device)
                full_content_tensor = data['content'].to(device) 
                content_img = full_content_tensor[:, 0:1, :, :] 
                
                # 即時萃取
                x1 = unet.mix_net.get_style_vectors(style_img, laplace_img).detach()
                
                # 計算 Flow Matching 損失
                b = x1.shape[0]
                x0 = torch.randn_like(x1)
                t = torch.rand(b, device=device)
                t_exp = t.view(b, 1, 1)
                x_t = (1 - t_exp) * x0 + t_exp * x1
                target_v = x1 - x0
                
                pred_v = vf_net(x_t, t, content_img=content_img)
                loss = torch.mean((pred_v - target_v) ** 2)
                
                val_loss += loss.item()
                val_steps += 1
                
        avg_val_loss = val_loss / val_steps
        
        # 紀錄 Epoch 結果
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | 🌟 Val Loss: {avg_val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6e}")

        # ------------------------------------------------------
        # [儲存機制]
        # ------------------------------------------------------
        # 如果當前 Epoch 的 Val Loss 是歷史最低，就儲存為 Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(opt.save_dir, "best_val_fm.pth")
            torch.save(vf_net.state_dict(), save_path)
            logger.info("  >> 🎉 發現更低的 Val Loss，已儲存最新最佳模型！")
            
        # 每隔 20 個 Epoch 額外儲存一個 Checkpoint 以備不時之需
        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(opt.save_dir, f"fm_epoch_{epoch+1}.pth")
            torch.save(vf_net.state_dict(), ckpt_path)
            logger.info(f"  >> 💾 已儲存定期備份: {ckpt_path}")

    logger.info(f"🎉 訓練完成！最佳 Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--unet_ckpt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./saved_flow_online')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--device', type=str, default='cuda')
    opt = parser.parse_args()
    main(opt)