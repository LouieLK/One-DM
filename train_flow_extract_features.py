"""
程式名稱: flow_extract_features.py
功能描述: 
    使用 h5py 動態寫入硬碟，解決大規模資料集 (380萬張) 訓練時的 OOM 記憶體耗盡問題。
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import h5py

# 引用專案模組
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from models.unet import UNetModel
from data_loader.loader import HandwritingDataset

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)

def main(args):
    logger = setup_logger()
    device = torch.device(args.device)

    # 自動將 .pt 替換為 .h5，讓您的 shell 腳本不用改也能無縫接軌
    save_path = args.save_path
    if save_path.endswith('.pt'):
        save_path = save_path.replace('.pt', '.h5')

    if os.path.exists(save_path):
        print(f"⚠️  警告: 檔案 {save_path} 已存在！")
        ans = input("是否要覆蓋？(y/n): ")
        if ans.lower() != 'y':
            print("已取消操作。")
            return

    logger.info(f"載入設定檔: {args.cfg}")
    cfg_from_file(args.cfg)
    assert_and_infer_cfg()

    logger.info(f"正在載入模型權重: {args.ckpt}")
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

    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    unet.load_state_dict(new_state_dict, strict=False)
    
    unet.eval() 
    unet.requires_grad_(False) 
    logger.info("✅ 模型載入完成且已凍結")

    HandwritingDataset.set_global_config(cfg)
    dataset = HandwritingDataset(split='train')
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True,
        collate_fn=dataset.collate_fn_
    )
    
    total_images = len(dataset)
    logger.info(f"📊 資料集總數: {total_images} 張圖片")
    logger.info(f"🚀 開始提取特徵並動態寫入 HDF5... (不再佔用龐大記憶體)")

    pbar = tqdm(loader, total=len(loader), desc="Extracting")
    # 在迴圈外先定義特徵維度
    flow_dim = cfg.MODEL.EMB_DIM * 2
    # 開啟 h5 檔案進行寫入
    with h5py.File(save_path, 'w') as f:
        # [修改] 1024 替換為 flow_dim
        dataset_h5 = f.create_dataset('features', shape=(0, flow_dim), maxshape=(None, flow_dim), dtype='float32', chunks=(2048, flow_dim))
        
        with torch.no_grad():
            for step, data in enumerate(pbar):
                style_img = data['style'].to(device)     
                laplace_img = data['laplace'].to(device) 
                
                low_vec, high_vec = unet.mix_net.get_style_vectors(style_img, laplace_img)
                features = torch.cat([low_vec, high_vec], dim=1)
                features_np = features.cpu().numpy()
                
                # 動態擴展硬碟空間並寫入
                curr_size = dataset_h5.shape[0]
                dataset_h5.resize(curr_size + features_np.shape[0], axis=0)
                dataset_h5[curr_size:] = features_np
                
                if step % 10 == 0:
                    pbar.set_postfix({"Cached Samples": dataset_h5.shape[0]})

    file_size_gb = os.path.getsize(save_path) / (1024**3)
    logger.info(f"🎉 成功完成！特徵已安全儲存。")
    logger.info(f"   - 總樣本數: {total_images}")
    logger.info(f"   - 檔案大小: {file_size_gb:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/chinese.yml')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='cached_features.pt')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)