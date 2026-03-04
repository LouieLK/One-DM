import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import logging # [新增] 引入 logging 模組

# 引入專案模組
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.util import fix_seed
from models.recognition import HTRNet

# ==========================================
# 0. Logger 設定
# ==========================================
def setup_logger(save_dir):
    logger = logging.getLogger("HTR_Train")
    logger.setLevel(logging.INFO)
    
    # 格式設定
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File Handler (寫入檔案)
    log_file = os.path.join(save_dir, 'train_htr.log')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Stream Handler (輸出到終端機)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

# ==========================================
# 1. Latent Dataset (含雜訊增強)
# ==========================================
class LatentHTRDataset(Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.root = cfg.DATASET.ROOT
        self.split = split
        
        self.letters = cfg.DATASET.LETTERS
        self.letter2index = {label: n + 1 for n, label in enumerate(self.letters)}
        
        self.latent_base_dir = os.path.join(self.root, cfg.DATASET.DIRS.LATENT)
        self.txt_path = os.path.join(self.root, cfg.DATASET.FILES[split])
        
        self.data_list = self.load_data(self.txt_path)
        print(f"[{split}] Loaded {len(self.data_list)} samples.")

    def load_data(self, data_path):
        data_list = []
        if not os.path.exists(data_path):
            print(f"[Warning] File not found: {data_path}")
            return []
            
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line: continue
                
                try:
                    parts = line.split(' ')
                    path_info = parts[0]
                    label = parts[1] if len(parts) > 1 else ""
                    
                    info = path_info.split(',')
                    s_id = info[0]
                    fname = info[1]
                    
                    latent_path = os.path.join(
                        self.latent_base_dir, 
                        self.split, 
                        s_id, 
                        fname + '.pt'
                    )
                    
                    data_list.append({
                        'path': latent_path,
                        'label': label
                    })
                except Exception as e:
                    continue
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        try:
            latent = torch.load(item['path'])
            
            # [新增] 雜訊增強 (Noise Augmentation)
            # 讓 HTR 模型適應有點雜訊的 Latent，這對之後的 Fine-tune 至關重要
            if self.split == 'train':
                # 加入 5% ~ 10% 的隨機雜訊
                noise_level = random.uniform(0.0, 0.1) 
                latent = latent + torch.randn_like(latent) * noise_level
                
        except Exception as e:
            latent = torch.zeros(4, 8, 8)
            
        label_str = item['label']
        label_indices = [self.letter2index[c] for c in label_str if c in self.letter2index]
        
        return latent, torch.IntTensor(label_indices), label_str

    def collate_fn(self, batch):
        latents = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        texts = [item[2] for item in batch]
        
        latents_batch = torch.stack(latents)
        labels_concat = torch.cat(labels)
        label_lengths = torch.IntTensor([len(l) for l in labels])
        
        return latents_batch, labels_concat, label_lengths, texts

# ==========================================
# 2. HTR Trainer (含 Logging)
# ==========================================
class HTRTrainer:
    def __init__(self, model, optimizer, train_loader, test_loader, device, cfg, logger):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.cfg = cfg
        self.logger = logger # [新增]
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        
        self.letters = cfg.DATASET.LETTERS
        self.index2letter = {n + 1: label for n, label in enumerate(self.letters)}
        
    def decode(self, preds):
        """Greedy Decode"""
        pred_indices = preds.argmax(2).permute(1, 0)
        decoded_strings = []
        for i in range(pred_indices.shape[0]):
            indices = pred_indices[i].tolist()
            result = []
            prev = -1
            for idx in indices:
                if idx != prev and idx != 0:
                    result.append(self.index2letter.get(idx, ''))
                prev = idx
            decoded_strings.append("".join(result))
        return decoded_strings

    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", ncols=100)
        total_loss = 0
        
        for step, (latents, labels, label_lengths, texts) in enumerate(pbar):
            latents = latents.to(self.device)
            labels = labels.to(self.device)
            label_lengths = label_lengths.to(self.device)
            
            preds = self.model(latents)
            
            # 維度調整
            real_batch_size = latents.size(0)
            if preds.size(0) == real_batch_size:
                preds = preds.permute(1, 0, 2)
            
            T = preds.size(0)
            B = preds.size(1)
            
            preds_log_softmax = preds.log_softmax(2)
            input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long).to(self.device)
            
            loss = self.criterion(preds_log_softmax, labels, input_lengths, label_lengths)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        # [新增] Log 紀錄
        self.logger.info(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")
        return avg_loss

    def test(self, epoch):
        self.model.eval()
        total_loss = 0
        pbar = tqdm(self.test_loader, desc="Testing", ncols=100)
        
        with torch.no_grad():
            for step, (latents, labels, label_lengths, texts) in enumerate(pbar):
                latents = latents.to(self.device)
                labels = labels.to(self.device)
                label_lengths = label_lengths.to(self.device)
                
                preds = self.model(latents)
                
                if preds.size(0) == latents.size(0):
                    preds = preds.permute(1, 0, 2)
                
                preds_log_softmax = preds.log_softmax(2)
                T, B = preds.size(0), preds.size(1)
                input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long).to(self.device)
                
                loss = self.criterion(preds_log_softmax, labels, input_lengths, label_lengths)
                total_loss += loss.item()
                
                # 每個 Epoch 隨機選一個 Batch 印出來看看
                if step == 0:
                    decoded = self.decode(preds)
                    log_msg = f"\n[Val Sample Epoch {epoch}]\nGT  : {texts[0]}\nPred: {decoded[0]}"
                    print(log_msg)
                    self.logger.info(log_msg.replace('\n', ' | ')) # 寫入 Log
                    
        avg_loss = total_loss / len(self.test_loader)
        self.logger.info(f"Epoch {epoch} | Test Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, path):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, path)
        self.logger.info(f"Saved Checkpoint: {path}")

# ==========================================
# 3. Main
# ==========================================
def main(opt):
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    fix_seed(cfg.TRAIN.SEED)
    
    # 準備輸出目錄
    save_dir = os.path.join(cfg.OUTPUT_DIR, 'htr_latent_model')
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化 Logger
    logger = setup_logger(save_dir)
    logger.info(f"Start Training HTR with Config: {opt.cfg_file}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    batch_size = 128 
    
    train_ds = LatentHTRDataset(cfg, split='train')
    test_ds = LatentHTRDataset(cfg, split='test')
    
    if len(train_ds) == 0:
        logger.error("[Error] 訓練集為空。")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_ds.collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=test_ds.collate_fn)
    
    n_classes = len(cfg.DATASET.LETTERS) + 1
    logger.info(f"Initializing HTRNet (Classes: {n_classes})...")
    
    model = HTRNet(nclasses=n_classes, vae=True, head='rnn', flattening='maxpool')
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    trainer = HTRTrainer(model, optimizer, train_loader, test_loader, device, cfg, logger)
    
    best_val_loss = float('inf')

    logger.info("Start Training Loop...")
    for epoch in range(101): # 建議跑 100 Epoch 左右即可
        loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} Loss: {loss:.4f}")
        
        if epoch > 1 and (epoch) % 5 == 0:
            val_loss = trainer.test(epoch)
            print(f"Epoch {epoch} Val Loss: {val_loss:.4f}")
            
            trainer.save_checkpoint(os.path.join(save_dir, f"htr_epoch_{epoch}.pth"))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_save_path = os.path.join(save_dir, "best_htr_model.pth")
                trainer.save_checkpoint(best_save_path)
                logger.info(f"★ New Best Model Saved! (Loss: {best_val_loss:.4f})")
                print(f"★ New Best Model Saved! (Loss: {best_val_loss:.4f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/chinese_ICADR2013_finetune.yml', help='Config file')
    opt = parser.parse_args()
    
    main(opt)