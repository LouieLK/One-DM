import argparse
import sys
import os
import yaml
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import AutoencoderKL

# 引入專案模組
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.util import fix_seed, load_specific_dict
from utils.logger import set_log
from data_loader.loader import HandwritingDataset
from trainer.trainer import Trainer
from models.unet import UNetModel
from models.diffusion import Diffusion
from models.loss import SupConLoss
from models.recognition import HTRNet

def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)
    """ prepare log file """
    logs = set_log(cfg.OUTPUT_DIR, opt.cfg_file, opt.log_name)

    """ set mulit-gpu """
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(opt.device, local_rank)
    
    HandwritingDataset.set_global_config(cfg)

    # ... (Dataset 載入部分保持您原本的設定) ...
    train_dataset = HandwritingDataset(split=cfg.TRAIN.TYPE)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.IMS_PER_BATCH, 
                                             sampler=train_sampler, num_workers=cfg.DATA_LOADER.NUM_THREADS,
                                             collate_fn=train_dataset.collate_fn_,
                                             pin_memory=True, drop_last=False)
    
    test_dataset = HandwritingDataset(split=cfg.TEST.TYPE)
    test_sampler = DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, 
                                            num_workers=cfg.DATA_LOADER.NUM_THREADS,
                                            collate_fn=test_dataset.collate_fn_)

    # ... (UNet 模型載入部分保持不變) ...
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, 
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
                     attention_resolutions=cfg.MODEL.ATTENTION_RESOLUTIONS, channel_mult=cfg.MODEL.CHANNEL_MULT, num_heads=cfg.MODEL.NUM_HEADS, 
                     context_dim=cfg.MODEL.EMB_DIM,
                    use_checkpoint=True   # 🌟 [關鍵新增] 開啟梯度檢查點！
                     ).to(device)

    # [重要] 載入 Stage 1 訓練好的 One-DM 權重
    if opt.one_dm != '':
        print(f"Loading Stage 1 One-DM weights from: {opt.one_dm}")
        ckpt = torch.load(opt.one_dm, map_location='cpu')
        # 處理可能的 module. 前綴 (如果是 DDP 存的)
        state_dict = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
        unet.load_state_dict(state_dict)

    # print("🚫 Force disabling gradient checkpointing for all modules...")
    # print("🥶 Freezing UNet Encoder (Input Blocks & Middle Block)...")
        
    # # 凍結 Input Blocks (Encoder)
    # for param in unet.input_blocks.parameters():
    #     param.requires_grad = False
        
    # # 凍結 Middle Block (橋接層)
    # for param in unet.middle_block.parameters():
    #     param.requires_grad = False

    # unet = DDP(unet, device_ids=[local_rank], broadcast_buffers=False,find_unused_parameters=True)    
    # optimizer = optim.AdamW(
    #     filter(lambda p: p.requires_grad, unet.parameters()), # 只更新沒被凍結的參數
    #     lr=cfg.SOLVER.BASE_LR
    # ) 
    unet = DDP(unet, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
    optimizer = optim.AdamW(unet.parameters(), lr=cfg.SOLVER.BASE_LR)
    ctc_loss = nn.CTCLoss()
    criterion = dict(nce=SupConLoss(contrast_mode='all'), recon=nn.MSELoss())
    diffusion = Diffusion(device=device, noise_offset=opt.noise_offset)

    '''load pretrained ocr model'''
    ocr_model = HTRNet(nclasses = len(cfg.DATASET.LETTERS) + 1, vae=True)
    if len(opt.ocr_model) > 0:
        miss, unxep = ocr_model.load_state_dict(torch.load(opt.ocr_model, map_location=torch.device('cpu')), strict=False)
        print('load pretrained ocr model from {}'.format(opt.ocr_model))
    else:
        print('failed to load the pretrained ocr model')
        exit()
    ocr_model.requires_grad_(False)
    ocr_model = ocr_model.to(device)
    
    """load pretrained vae"""
    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    vae = vae.to(device)


    """build trainer"""
    trainer = Trainer(diffusion, unet, vae, criterion, optimizer, train_loader, logs, test_loader, device, ocr_model, ctc_loss)
    trainer.train()

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--cfg', dest='cfg_file', default='configs/chinese_finetune.yml')
    parser.add_argument('--one_dm', dest='one_dm', required=True, help='Path to Stage 1 checkpoint')
    parser.add_argument('--ocr_model', dest='ocr_model', required=True, help='Path to converted OCR .pth')
    parser.add_argument('--log', default='finetune_log', dest='log_name')
    parser.add_argument('--noise_offset', default=0, type=float)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--local_rank', type=int, default=0)
    opt = parser.parse_args()
    main(opt)