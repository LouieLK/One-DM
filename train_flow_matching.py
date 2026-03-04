"""
程式名稱: train_flow_matching.py
功能描述: 
    讀取 HDF5 特徵檔來訓練 Flow Matching 模型。
    (已加入 Chunked Loading 避免 OOM)
"""
import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
import time
import h5py

from models.flow_matching import VectorFieldNetwork

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
    
    # 自動檔名修正
    cached_file = opt.cached_file
    if cached_file.endswith('.pt'):
        cached_file = cached_file.replace('.pt', '.h5')
        
    logger.info("🚀 啟動極速 Flow Matching 訓練")
    logger.info(f"📂 正在讀取特徵檔: {cached_file} ...")
    t0 = time.time()
    
    if not os.path.exists(cached_file):
        raise FileNotFoundError(f"找不到 {cached_file}，請先執行 flow_extract_features.py！")

    # -----------------------------------------------------------
    # 安全且極速的記憶體載入法 (Chunked Loading) - 預防 Killed
    # -----------------------------------------------------------
    with h5py.File(cached_file, 'r') as f:
        num_samples = f['features'].shape[0]
        dim = f['features'].shape[1]
        
        logger.info(f"   - 準備在 RAM 預分配空間 ({num_samples} x {dim})...")
        features = torch.empty((num_samples, dim), dtype=torch.float32)
        
        chunk_size = 100000
        for i in tqdm(range(0, num_samples, chunk_size), desc="Loading to RAM"):
            end = min(i + chunk_size, num_samples)
            features[i:end] = torch.from_numpy(f['features'][i:end])
            
    logger.info(f"✅ 載入完成！耗時 {time.time()-t0:.2f} 秒")
    logger.info(f"   - 樣本總數: {features.shape[0]}")

    dataset = TensorDataset(features)
    train_loader = DataLoader(
        dataset, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )

    vf_net = VectorFieldNetwork(in_dim=1024, hidden_dim=1024, num_layers=8).to(device)
    optimizer = optim.AdamW(vf_net.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    best_loss = float('inf')
    logger.info("🔥 開始訓練...")

    for epoch in range(opt.epochs):
        vf_net.train()
        epoch_loss = 0.0
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epochs}")
        
        for (x1_batch,) in pbar:
            x1 = x1_batch.to(device)
            b = x1.shape[0]
            
            x0 = torch.randn_like(x1)
            t = torch.rand(b, device=device)
            t_expand = t.view(-1, 1)
            x_t = (1 - t_expand) * x0 + t_expand * x1
            target_v = x1 - x0
            
            pred_v = vf_net(x_t, t)
            loss = torch.mean((pred_v - target_v) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vf_net.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        avg_loss = epoch_loss / steps
        logger.info(f"Epoch {epoch+1} Done | Avg Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(opt.save_dir, "best_fm.pth")
            torch.save(vf_net.state_dict(), save_path)
            
        if (epoch + 1) % 10 == 0:
            periodic_path = os.path.join(opt.save_dir, f"fm_epoch_{epoch+1}.pth")
            torch.save(vf_net.state_dict(), periodic_path)

    logger.info(f"🎉 訓練結束！最佳 Loss: {best_loss:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cached_file', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./saved_flow_matching')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8192) 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    opt = parser.parse_args()
    main(opt)