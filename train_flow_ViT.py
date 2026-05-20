"""
程式名稱: train_flow_ViT.py
功能描述: 
    讀取 HDF5 特徵檔來訓練 Flow Matching 模型。
    (已升級: SOTA 內容條件化 DiT、空間對齊注入、支援楷體特徵引導)
"""
import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import logging
import time
import h5py

# 引入 One-DM 的設定與 Content 讀取器
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from data_loader.loader import ContentData

# 🌟 引入我們升級版的 SOTA DiT 模型
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

# ==========================================================
# 🌟 Lazy Loading Dataset (新增讀取標籤功能)
# ==========================================================
class H5FeatureDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as f:
            self.num_samples = f['features'].shape[0]
            self.seq_len = f['features'].shape[1]
            self.dim = f['features'].shape[2]
            
            # ⚠️ 注意：您的 HDF5 必須有一個 'labels' dataset 存放每個特徵對應的字！
            if 'labels' not in f:
                raise ValueError("❌ HDF5 檔案中找不到 'labels'！訓練 Content-Conditioned 模型必須知道特徵對應哪個字。")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not hasattr(self, 'h5_file'):
            self.h5_file = h5py.File(self.h5_path, 'r')
            
        feat = self.h5_file['features'][idx]
        
        # 讀取標籤 (處理 byte string 與普通 string 的差異)
        label = self.h5_file['labels'][idx]
        if isinstance(label, bytes):
            label = label.decode('utf-8')
            
        return torch.from_numpy(feat), label

# ==========================================================
# 🚀 主程式
# ==========================================================
def main(opt):
    device = torch.device(opt.device)
    logger = setup_logger(opt.save_dir)
    
    # 載入設定檔 (取得 128x128 圖片設定)
    cfg_from_file(opt.cfg)
    assert_and_infer_cfg()
    
    cached_file = opt.cached_file
    if cached_file.endswith('.pt'):
        cached_file = cached_file.replace('.pt', '.h5')
        
    logger.info("🚀 啟動 Content-Conditioned DiT 訓練 (空間對齊注入)")
    
    # 🌟 載入 Content 讀取器
    content_loader = ContentData(split='train', content_type='kaifont', cfg=cfg)

    dataset = H5FeatureDataset(cached_file)
    seq_len = dataset.seq_len
    dim = dataset.dim
    
    val_size = int(0.02 * dataset.num_samples)
    train_size = dataset.num_samples - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    logger.info(f"✅ 資料集掛載完成！訓練: {train_size}, 驗證: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化 DiT 模型
    vf_net = VectorFieldDiT(seq_len=seq_len, in_dim=dim, hidden_dim=1024, num_layers=8, num_heads=16).to(device)
    optimizer = optim.AdamW(vf_net.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    best_val_loss = float('inf')

    # --- 輔助函數：將 Batch 的字元轉為 Tensor 圖片 ---
    def get_content_batch(chars):
        c_refs = []
        for c in chars:
            c_img = content_loader.get_content(c) # [C, H, W]
            while c_img.dim() > 3: c_img = c_img.squeeze(0)
            while c_img.dim() < 3: c_img = c_img.unsqueeze(0)
            c_refs.append(c_img)
        return torch.stack(c_refs).to(device)

    for epoch in range(opt.epochs):
        # ------------------------------------------------------
        # 1. 訓練階段 (Train)
        # ------------------------------------------------------
        vf_net.train()
        epoch_loss = 0.0
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epochs} [Train]")
        
        # DataLoader 現在會回傳特徵與對應的字串列表
        for x1_batch, chars in pbar:
            x1 = x1_batch.to(device) 
            b = x1.shape[0]
            
            # 🌟 動態獲取這個 Batch 的楷體骨架圖片
            content_img = get_content_batch(chars) # [B, 1, 128, 128]
            
            x0 = torch.randn_like(x1)
            t = torch.rand(b, device=device)
            t_expand = t.view(b, 1, 1)
            
            x_t = (1 - t_expand) * x0 + t_expand * x1
            target_v = x1 - x0
            
            # 🌟 餵入含有骨架的特徵！
            pred_v = vf_net(x_t, t, content_img=content_img)
            
            loss = torch.mean((pred_v - target_v) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vf_net.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        avg_train_loss = epoch_loss / steps

        # ------------------------------------------------------
        # 2. 驗證階段 (Validation)
        # ------------------------------------------------------
        vf_net.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for x1_batch, chars in tqdm(val_loader, desc=f"Epoch {epoch+1}/{opt.epochs} [Val]", leave=False):
                x1 = x1_batch.to(device)
                b = x1.shape[0]
                content_img = get_content_batch(chars)
                
                x0 = torch.randn_like(x1)
                t = torch.rand(b, device=device)
                t_expand = t.view(b, 1, 1)
                
                x_t = (1 - t_expand) * x0 + t_expand * x1
                target_v = x1 - x0
                
                pred_v = vf_net(x_t, t, content_img=content_img)
                val_loss += torch.mean((pred_v - target_v) ** 2).item()
                val_steps += 1
                
        avg_val_loss = val_loss / val_steps
        logger.info(f"Epoch {epoch+1} Done | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # ------------------------------------------------------
        # 3. 儲存機制與預覽
        # ------------------------------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(vf_net.state_dict(), os.path.join(opt.save_dir, "best_val_fm.pth"))
            logger.info("  >> 🎉 已儲存最新最佳模型！")
            
        if (epoch + 1) % 20 == 0:
            logger.info("  >> 🔍 正在進行特徵採樣測試 (Preview)...")
            with torch.no_grad():
                # 測試固定字元，看看模型是否真能照著畫
                test_chars = ["永", "霸", "鱗", "龜"] * 4 # 產生 16 張圖
                test_content_img = get_content_batch(test_chars)
                
                z_test = torch.randn(16, seq_len, dim, device=device)
                # 🌟 採樣時加入骨架引導！
                preview_features = sample_flow_matching(vf_net, z_test, content_img=test_content_img, steps=20)
                
                preview_path = os.path.join(opt.save_dir, f"preview_features_epoch_{epoch+1}.pt")
                torch.save(preview_features.cpu(), preview_path)
            
            torch.save(vf_net.state_dict(), os.path.join(opt.save_dir, f"fm_epoch_{epoch+1}.pth"))

    logger.info(f"🎉 訓練結束！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/chinese_CASIA-HWDB1.0-1.1.yml')
    parser.add_argument('--cached_file', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./saved_flow_matching')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=256) 
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    opt = parser.parse_args()
    main(opt)