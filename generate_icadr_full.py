import argparse
import os
import torch
import cv2
import numpy as np
import datetime
import random
from pathlib import Path
from tqdm import tqdm
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
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
# 🛠️ 輔助工具：Grid 佈局 (用於視覺化)
# ==========================================
class GridLayout:
    def __init__(self, cols_config, height=64, sep_width=2, bg_color=255):
        self.cols = cols_config
        self.height = height
        self.sep_width = sep_width
        self.bg_color = bg_color
        self.total_width = sum([c['width'] for c in self.cols]) + (len(self.cols) - 1) * sep_width

    def draw_header(self, font_scale=0.5, thickness=1):
        header_h = 50 
        header_img = np.ones((header_h, self.total_width, 3), dtype=np.uint8) * 240 
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_scale = 0.35
        (ts_w, ts_h), _ = cv2.getTextSize(now_str, font, ts_scale, 1)
        cv2.putText(header_img, now_str, (self.total_width - ts_w - 5, 15), font, ts_scale, (100,100,100), 1)

        current_x = 0
        for i, col in enumerate(self.cols):
            text = col['name']
            w = col['width']
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = int(current_x + (w - text_w) / 2)
            text_y = int((header_h + text_h) / 2) + 2 
            
            color = col.get('color', (0, 0, 0))
            cv2.putText(header_img, text, (text_x, text_y), font, font_scale, color, thickness)
            current_x += w + self.sep_width
        return header_img

    def stitch_row(self, images):
        row_img = np.ones((self.height, self.total_width, 3), dtype=np.uint8) * self.bg_color
        current_x = 0
        sep_col = np.ones((self.height, self.sep_width, 3), dtype=np.uint8) * 200 
        
        for i, col in enumerate(self.cols):
            key = col.get('key')
            w = col['width']
            
            if key in images and images[key] is not None:
                img = images[key]
                if img.shape[0] != self.height or img.shape[1] != w:
                    img = cv2.resize(img, (w, self.height))
                row_img[:, current_x:current_x+w] = img
            else:
                if 'text' in images and key == 'id':
                    white_block = np.ones((self.height, w, 3), dtype=np.uint8) * 255
                    id_str = str(images['text'])
                    (id_w, id_h), _ = cv2.getTextSize(id_str, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    id_x = int((w - id_w) / 2)
                    id_y = int((self.height + id_h) / 2)
                    cv2.putText(white_block, id_str, (id_x, id_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
                    row_img[:, current_x:current_x+w] = white_block

            current_x += w
            if i < len(self.cols) - 1:
                row_img[:, current_x:current_x+self.sep_width] = sep_col
                current_x += self.sep_width
        return row_img

def tensor_to_cv2(tensor):
    t = tensor.detach().cpu()
    if t.min() < 0: t = t * 0.5 + 0.5
    t = t.clamp(0, 1)
    img_np = t.permute(1, 2, 0).numpy() * 255
    img_np = img_np.astype(np.uint8)
    if img_np.shape[2] == 1:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np

def tensor_to_save(tensor):
    t = tensor.detach().cpu()
    if t.min() < 0: t = t * 0.5 + 0.5
    t = t.clamp(0, 1)
    img_np = t.permute(1, 2, 0).numpy() * 255
    img_np = img_np.astype(np.uint8)
    if img_np.shape[2] == 1:
        img_np = img_np.squeeze(2) 
    return img_np

# ==========================================
# 🚀 主程式
# ==========================================
def main(args):
    fix_seed(args.seed)
    cfg_from_file(args.cfg)
    assert_and_infer_cfg()
    device = torch.device(args.device)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Generating Full Dataset (Reconstruction & Flow Substitution)")
    print(f"📂 Output Dir: {output_dir}")

    # 1. 載入模型
    print("Loading Models...")
    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    
    unet = UNetModel(
        in_channels=cfg.MODEL.IN_CHANNELS, 
        model_channels=cfg.MODEL.EMB_DIM, 
        out_channels=cfg.MODEL.OUT_CHANNELS, 
        num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
        attention_resolutions=cfg.MODEL.ATTENTION_RESOLUTIONS, 
        channel_mult=cfg.MODEL.CHANNEL_MULT, 
        num_heads=cfg.MODEL.NUM_HEADS, 
        context_dim=cfg.MODEL.EMB_DIM
    ).to(device)

    print(f"Loading weights from: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    unet.load_state_dict(new_state_dict, strict=False)
    unet.eval()
    
    diffusion = Diffusion(device=device)

    # === [新增] 載入 Flow 模型 ===
    flow_model = None
    if args.flow_type and args.flow_ckpt:
        print(f"Loading Flow Model ({args.flow_type}) from: {args.flow_ckpt}")
        if args.flow_type == 'fm':
            flow_model = VectorFieldNetwork(in_dim=1024, hidden_dim=1024, num_layers=8).to(device)
        elif args.flow_type == 'joint':
            flow_model = NormalizingFlow(num_inputs=1024, num_hidden=2048, num_layers=16).to(device)
        else:
            raise ValueError(f"Unknown flow_type: {args.flow_type}")
        
        flow_ckpt = torch.load(args.flow_ckpt, map_location=device)
        flow_model.load_state_dict(flow_ckpt)
        flow_model.eval()
    # ==============================

    # 2. 準備資料集
    HandwritingDataset.set_global_config(cfg)
    test_dataset = HandwritingDataset(split=args.split) # 這裡可以是 train 或 test
    content_loader = ContentData(split='train', content_type='unifont', cfg=cfg)
    
    total_images = len(test_dataset)
    print(f"✅ Dataset Loaded. Split: {args.split}, Total images: {total_images}")
    
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 3. 設定視覺化採樣
    random.seed(args.seed)
    viz_indices = set(random.sample(range(total_images), min(args.viz_num, total_images)))
    print(f"📸 Selected {len(viz_indices)} samples for visualization grid.")
    
    grid_config = [
        {'name': 'ID',      'width': 60, 'key': 'id'},
        {'name': 'Style',   'width': 64, 'key': 'style'},
        {'name': 'Content', 'width': 64, 'key': 'content'},
        {'name': 'Real GT', 'width': 64, 'key': 'gt'},
        {'name': 'Ours',    'width': 64, 'key': 'pred', 'color': (0, 0, 200)}
    ]
    layout = GridLayout(grid_config, height=64)
    viz_rows = []

    success_count = 0
    skip_count = 0
    global_dataset_idx = 0
    
    existing_counts = defaultdict(lambda: defaultdict(int))
    
    # === [新增] Flow Style 快取字典 ===
    # 用來確保同一個 writer 永遠對應到同一個 Flow 隨機生成的風格向量
    flow_style_cache = {}
    # ==============================
    
    print("⚡ Starting Generation...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            current_batch_size = len(batch['wid'])
            
            style_imgs = batch['style'][:, 0:1, :, :].to(device)
            laplace_imgs = batch['laplace'][:, 0:1, :, :].to(device)
            gt_imgs = batch['img'].to(device) 
            contents = batch['content']
            
            valid_batch_indices = []
            content_ref_list = []
            
            # 若啟用 Flow，則準備存放 1D 向量
            if flow_model is not None:
                style_list = []
                laplace_list = []
            
            # === [新增] 斷點續傳邏輯的準備 ===
            # 先檢查這整個 Batch 的圖片是否都已經存在了
            need_to_generate_flags = []
            
            for i, char_text in enumerate(contents):
                abs_dataset_idx = global_dataset_idx + i
                original_record = test_dataset.data_dict[test_dataset.indices[abs_dataset_idx]]
                str_wid = str(original_record['s_id'])
                
                char = char_text
                count = existing_counts[str_wid][char]
                fname = f"{char}.png" if count == 0 else f"{char}_{count}.png"
                
                writer_dir = output_dir / str_wid
                save_path = writer_dir / fname
                
                # 如果檔案已經存在，標記為 False (不需要生成)
                if save_path.exists():
                    need_to_generate_flags.append(False)
                    # 計數器還是要推進，確保後面的檔案命名正確
                    existing_counts[str_wid][char] += 1
                else:
                    need_to_generate_flags.append(True)
            
            # 只有當這個 Batch 裡有「需要生成」的圖片時，才進行複雜的網路推論
            if any(need_to_generate_flags):
                for i, char_text in enumerate(contents):
                    # 如果這張圖已經存在，跳過網路推論的準備
                    if not need_to_generate_flags[i]:
                        continue
                        
                    try:
                        c_ref = content_loader.get_content(char_text)
                        
                        abs_dataset_idx = global_dataset_idx + i
                        original_record = test_dataset.data_dict[test_dataset.indices[abs_dataset_idx]]
                        str_wid = str(original_record['s_id'])
                        
                        if flow_model is not None:
                            if str_wid not in flow_style_cache:
                                z = torch.randn(1, 1024).to(device)
                                if args.flow_type == 'fm':
                                    feat = sample_flow_matching(flow_model, z)
                                elif args.flow_type == 'joint':
                                    feat = flow_model.reverse(z)
                                flow_style_cache[str_wid] = feat
                            
                            cached_feat = flow_style_cache[str_wid]
                            style_list.append(cached_feat[0, :512])
                            laplace_list.append(cached_feat[0, 512:])

                        content_ref_list.append(c_ref)
                        valid_batch_indices.append(i)
                    except Exception:
                        skip_count += 1
                        continue
                
                if not content_ref_list:
                    global_dataset_idx += current_batch_size
                    continue

                content_ref_batch = torch.cat(content_ref_list, dim=0).to(device)
                
                if flow_model is not None:
                    style_batch = torch.stack(style_list).to(device)
                    laplace_batch = torch.stack(laplace_list).to(device)
                    h_latent = 8 
                    w_latent = 8
                else:
                    style_batch = style_imgs[valid_batch_indices]
                    laplace_batch = laplace_imgs[valid_batch_indices]
                    h_latent = style_batch.shape[2] // 8
                    w_latent = (content_ref_batch.shape[1] * 64) // 8 
                
                # === [修復] 定義 gt_batch，供視覺化使用 ===
                gt_batch = gt_imgs[valid_batch_indices]
                # ============================================

                noise = torch.randn((len(valid_batch_indices), 4, h_latent, w_latent)).to(device)
                
                preds = diffusion.ddim_sample(
                    model=unet, vae=vae, n=len(valid_batch_indices), 
                    x=noise, styles=style_batch, laplace=laplace_batch, 
                    content=content_ref_batch, sampling_timesteps=50
                )
                
                for k, idx in enumerate(valid_batch_indices):
                    abs_dataset_idx = global_dataset_idx + idx
                    
                    original_record = test_dataset.data_dict[test_dataset.indices[abs_dataset_idx]]
                    str_wid = str(original_record['s_id'])
                    
                    char = contents[idx]
                    count = existing_counts[str_wid][char]
                    fname = f"{char}.png" if count == 0 else f"{char}_{count}.png"
                    existing_counts[str_wid][char] += 1
                    
                    writer_dir = output_dir / str_wid
                    writer_dir.mkdir(parents=True, exist_ok=True)
                    save_path = writer_dir / fname
                    
                    img_save = tensor_to_save(preds[k])
                    if len(img_save.shape) == 3 and img_save.shape[2] == 3:
                        img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(save_path), img_save)
                    
                    success_count += 1
                    
                    if abs_dataset_idx in viz_indices:
                        if flow_model is not None:
                            viz_style = np.ones((64, 64, 3), dtype=np.uint8) * 230
                            cv2.putText(viz_style, "Flow", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)
                        else:
                            viz_style = tensor_to_cv2(style_batch[k])
                            
                        viz_content = tensor_to_cv2(content_ref_batch[k])
                        viz_gt = tensor_to_cv2(gt_batch[k]) # 這裡就不會再報錯了！
                        viz_pred = tensor_to_cv2(preds[k])
                        
                        row_data = {
                            'id': None, 
                            'text': str_wid,
                            'style': viz_style,
                            'content': viz_content,
                            'gt': viz_gt,
                            'pred': viz_pred
                        }
                        final_row = layout.stitch_row(row_data)
                        viz_rows.append(final_row)
            else:
                # 整個 Batch 都存在，直接跳過 (但記得推進 dataset_idx)
                pass
                
            global_dataset_idx += current_batch_size

    # 最後輸出 Grid 圖
    # if viz_rows:
    #     print(f"🎨 Stitching {len(viz_rows)} comparison samples...")
    #     header_img = layout.draw_header()
    #     grid_img = np.vstack([header_img] + viz_rows)
        
    #     timestamp_fname = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     viz_save_path = output_dir / f"summary_grid_{timestamp_fname}.png"
    #     cv2.imwrite(str(viz_save_path), grid_img)
    #     print(f"✨ Visualization saved to: {viz_save_path}")

    print(f"🎉 All Done!")
    print(f"   Success: {success_count}, Skipped: {skip_count}")
    print(f"   Flow Styles Cached: {len(flow_style_cache)} unique virtual writers.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/chinese.yml')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results/icdar2013_test_generated')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--viz_num', type=int, default=20, help="要隨機抽取多少張圖進行視覺化比對")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--split', type=str, default='test', help="指定要覆寫 train 還是 test")
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    
    # --- 新增 Flow 相關參數 ---
    parser.add_argument('--flow_type', type=str, choices=['fm', 'joint'], default=None, 
                        help="Specify which flow model to use: 'fm' or 'joint'")
    parser.add_argument('--flow_ckpt', type=str, default=None, 
                        help="Path to the trained flow model checkpoint.")
    # --------------------------
    
    args = parser.parse_args()
    main(args)