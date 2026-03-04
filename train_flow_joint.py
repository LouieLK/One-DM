"""
程式名稱: train_flow_joint.py
功能描述: 
    使用預先提取的特徵檔 (.h5) 來訓練 Joint Normalizing Flow 模型。
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

from models.normalizing_flow import NormalizingFlow

def setup_logger(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = os.path.join(save_dir, "train_joint_flow_log.txt")
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
        
    logger.info("🚀 啟動 Joint Flow 極速訓練")
    logger.info(f"📂 正在讀取特徵檔: {cached_file} ...")
    t0 = time.time()
    
    if not os.path.exists(cached_file):
        raise FileNotFoundError(f"找不到 {cached_file}！請先執行 flow_extract_features.py")

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
    
    dataset = TensorDataset(features)
    train_loader = DataLoader(
        dataset, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )

    input_dim = 1024 
    logger.info(f"Building Joint Normalizing Flow (Dim: {input_dim})...")
    
    flow_model = NormalizingFlow(num_inputs=input_dim, num_hidden=input_dim*2, num_layers=16).to(device)
    
    optimizer = optim.Adam(flow_model.parameters(), lr=opt.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    logger.info(f"Start training Joint Flow for {opt.epochs} epochs...")
    best_loss = float('inf')

    for epoch in range(opt.epochs):
        flow_model.train()
        total_loss = 0
        count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epochs}")
        
        for (s_joint,) in pbar:
            s_joint = s_joint.to(device)

            z, log_det_jacobian = flow_model(s_joint)
            
            log_prob_z = -0.5 * torch.sum(z**2, dim=1)
            log_likelihood = log_prob_z + log_det_jacobian
            loss = -torch.mean(log_likelihood)
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        avg_loss = total_loss / count
        logger.info(f"Epoch {epoch+1} | NLL Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(flow_model.state_dict(), os.path.join(opt.save_dir, "best_joint_flow.pth"))
            
        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(opt.save_dir, f"ckpt_epoch_{epoch+1}.pth")
            torch.save(flow_model.state_dict(), os.path.join(ckpt_path))

    logger.info(f"🎉 訓練結束！最佳 Loss: {best_loss:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cached_file', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./saved_joint_flow')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    
    opt = parser.parse_args()
    main(opt)