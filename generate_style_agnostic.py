import argparse
import os
import torch
import cv2
import numpy as np
import random
import math
from pathlib import Path
from tqdm import tqdm
from diffusers import AutoencoderKL
from collections import defaultdict

# 引用 One-DM 模組
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from models.unet import UNetModel
from models.diffusion import Diffusion
from utils.util import fix_seed
from data_loader.loader import HandwritingDataset, ContentData

# --- 新增 Flow 模型引用 ---
from models.flow_matching import VectorFieldNetwork, sample_flow_matching
from models.normalizing_flow import NormalizingFlow
# --------------------------

# ==========================================
# 🛠️ 輔助工具 (直接複製自 generate_icadr_full.py)
# ==========================================
def tensor_to_save(tensor):
    """轉為原始格式供存檔 (保留灰階/RGB)"""
    t = tensor.detach().cpu()
    if t.min() < 0: t = t * 0.5 + 0.5
    t = t.clamp(0, 1)
    img_np = t.permute(1, 2, 0).numpy() * 255
    img_np = img_np.astype(np.uint8)
    if img_np.shape[2] == 1:
        img_np = img_np.squeeze(2) 
    return img_np

def main(args):
    # 1. 初始化配置
    fix_seed(args.seed)
    cfg_from_file(args.cfg)
    assert_and_infer_cfg()
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 初始化 Style-Agnostic 生成 (Content 平衡版)")
    print(f"🎯 目標生成總數: {args.total_num}")
    print(f"📂 輸出目錄: {output_dir}")

    # 2. 載入模型
    print("Loading Models...")
    
    # VAE
    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    
    # UNet
    unet = UNetModel(
        in_channels=cfg.MODEL.IN_CHANNELS, 
        model_channels=cfg.MODEL.EMB_DIM, 
        out_channels=cfg.MODEL.OUT_CHANNELS, 
        num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
        attention_resolutions=cfg.MODEL.ATTENTION_RESOLUTIONS, 
        channel_mult=cfg.MODEL.CHANNEL_MULT, 
        num_heads=cfg.MODEL.NUM_HEADS, 
        context_dim=cfg.MODEL.EMB_DIM,
        backbone_type=args.backbone_type
    ).to(device)

    # 載入 UNet Checkpoint
    print(f"Loading UNet weights from: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    unet.load_state_dict(new_state_dict, strict=False)
    unet.eval()
    
    diffusion = Diffusion(device=device, noise_offset=0.0)

    # === [新增] 載入 Flow 模型 ===
    flow_model = None
    if args.flow_type and args.flow_ckpt:
        print(f"Loading Flow Model ({args.flow_type}) from: {args.flow_ckpt}")
        if args.flow_type == 'fm': # Flow Matching
            flow_model = VectorFieldNetwork(in_dim=1024, hidden_dim=1024, num_layers=8).to(device)
        elif args.flow_type == 'joint': # Joint Flow
            flow_model = NormalizingFlow(num_inputs=1024, num_hidden=2048, num_layers=16).to(device)
        else:
            raise ValueError(f"Unknown flow_type: {args.flow_type}")
        
        flow_ckpt = torch.load(args.flow_ckpt, map_location=device)
        flow_model.load_state_dict(flow_ckpt)
        flow_model.eval()
    # ==============================

    # 3. 準備資料集
    HandwritingDataset.set_global_config(cfg)
    
    # 如果不使用 Flow，我們需要 style_dataset 來提供參考圖像
    style_dataset = None
    available_indices = []
    if flow_model is None:
        style_dataset = HandwritingDataset(split='test')
        available_indices = style_dataset.indices
    
    content_loader = ContentData(split='train', content_type='kaifont', cfg=cfg)

    # 4. 規劃 Content 平衡任務
    print("📋 Planning balanced tasks...")
    tasks = []
    existing_counts = defaultdict(lambda: defaultdict(int)) 
    
    all_chars_str = cfg.DATASET.LETTERS 
    all_chars_pool = list(all_chars_str) 
    num_chars = len(all_chars_pool)
    
    repeats = math.ceil(args.total_num / num_chars) if num_chars > 0 else 1
    balanced_char_list = all_chars_pool * repeats
    balanced_char_list = balanced_char_list[:args.total_num]
    random.shuffle(balanced_char_list)
    
    print(f"📊 Content Balancing: {num_chars} unique chars, generating {len(balanced_char_list)} images.")
    
    for char in balanced_char_list:
        if flow_model is None:
            # 模式 1: 使用真實參考圖像，隨機選擇一個 Writer
            rand_style_idx = random.randint(0, len(available_indices) - 1)
            data_key = available_indices[rand_style_idx]
            wid = str(style_dataset.data_dict[data_key]['s_id'])
        else:
            # 模式 2: 使用 Flow，不需要真實 Writer ID，生成一個虛擬 ID
            wid = f"flow_style_{random.randint(0, 9999):04d}"
            
        count = existing_counts[wid][char]
        fname = f"{char}.png" if count == 0 else f"{char}_{count}.png"
        existing_counts[wid][char] += 1
        
        task_info = {'wid': wid, 'char': char, 'fname': fname}
        if flow_model is None:
            task_info['style_idx'] = rand_style_idx
            
        tasks.append(task_info)

    # 5. 執行生成
    batch_size = args.batch_size
    num_batches = (len(tasks) + batch_size - 1) // batch_size
    print(f"🚀 Start Generating {len(tasks)} images in {num_batches} batches...")

    success_count = 0
    skip_count = 0

    for i in tqdm(range(num_batches), desc="Generating"):
        current_tasks = tasks[i*batch_size : (i+1)*batch_size]
        
        valid_indices = []
        content_ref_list = []
        
        # 收集 Content Data
        for k, task in enumerate(current_tasks):
            try:
                c_ref = content_loader.get_content(task['char']) 
                content_ref_list.append(c_ref)
                valid_indices.append(k)
            except Exception:
                skip_count += 1

        if not valid_indices:
            continue

        content_batch = torch.cat(content_ref_list, dim=0).to(device)
        B = len(valid_indices)

        with torch.no_grad():
            # === [新增] 準備 Style Features ===
            if flow_model is not None:
                # 模式 2: 使用 Flow 生成特徵向量
                z = torch.randn(B, 1024).to(device)
                
                if args.flow_type == 'fm':
                    generated_features = sample_flow_matching(flow_model, z)
                elif args.flow_type == 'joint':
                    generated_features = flow_model.reverse(z)
                
                # 假設前 512 維是 low_vec，後 512 維是 high_vec
                # (這取決於您在 flow_extract_features.py 中 torch.cat 的順序)
                # 您先前的代碼是: features = torch.cat([low_vec, high_vec], dim=1)
                style_batch = generated_features[:, :512]   # 對應 anchor_low_feature
                laplace_batch = generated_features[:, 512:] # 對應 anchor_high_feature
                
            else:
                # 模式 1: 使用真實參考圖像 (與原本邏輯相同)
                style_list = []
                laplace_list = []
                for valid_idx in valid_indices:
                    task = current_tasks[valid_idx]
                    style_ref, laplace_ref = style_dataset.get_style_ref(task['wid'])
                    style_list.append(torch.from_numpy(style_ref).to(torch.float32))
                    laplace_list.append(torch.from_numpy(laplace_ref).to(torch.float32))
                
                style_batch = torch.stack(style_list).to(device)
                laplace_batch = torch.stack(laplace_list).to(device)
            # ==============================

            # 初始雜訊
            # 注意: 如果是 Flow 模式，style_batch 只是 1D 向量，無法用 .shape[2] 獲取空間維度
            # 因此，我們需要直接指定空間維度。對於 64x64 的圖像，UNet 在經過 downsampling 後，
            # latent resolution 通常是 64 // 8 = 8。
            h_latent = 8 
            w_latent = 8
            noise = torch.randn((B, 4, h_latent, w_latent)).to(device)

            # Diffusion Sampling
            # 這裡我們將 vae 傳入 ddim_sample，依照 generate_icadr_full.py 的做法，它會回傳解碼後的圖片
            preds = diffusion.ddim_sample(
                model=unet, 
                vae=vae, 
                n=B, 
                x=noise, 
                styles=style_batch, # 如果用 Flow，這裡是 [B, 512]；如果用圖片，這裡是 [B, 2, H, W]
                laplace=laplace_batch, # 同上
                content=content_batch, 
                sampling_timesteps=50
            )
            
        # 存檔
        for k, valid_idx in enumerate(valid_indices):
            task = current_tasks[valid_idx]
            writer_dir = output_dir / task['wid']
            writer_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = writer_dir / task['fname']
            img_save = tensor_to_save(preds[k])
            
            if len(img_save.shape) == 3 and img_save.shape[2] == 3:
                img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)
                
            cv2.imwrite(str(save_path), img_save)
            success_count += 1

    print(f"🎉 Generation Complete!")
    print(f"   Success: {success_count}, Skipped: {skip_count}")
    print(f"   Saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/chinese.yml')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to UNet checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results/style_agnostic')
    parser.add_argument('--total_num', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--backbone_type', type=str, default='resnet')
    
    # --- 新增 Flow 相關參數 ---
    parser.add_argument('--flow_type', type=str, choices=['fm', 'joint'], default=None, 
                        help="Specify which flow model to use: 'fm' for Flow Matching, 'joint' for Joint Flow.")
    parser.add_argument('--flow_ckpt', type=str, default=None, 
                        help="Path to the trained flow model checkpoint.")
    # --------------------------

    args = parser.parse_args()
    main(args)