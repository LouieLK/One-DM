import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.util import fix_seed
from utils.logger import set_log
from data_loader.loader import HandwritingDataset 
import torch
from trainer.trainer import Trainer
from models.unet import UNetModel
from torch import optim
import torch.nn as nn
from models.diffusion import Diffusion, EMA
from diffusers import AutoencoderKL
from models.loss import SupConLoss


def main(opt):
    # 🌟 [新增] 解鎖 TF32 算力 (非常重要)
    torch.set_float32_matmul_precision('high')
    
    # 🌟 [新增] 強制 PyTorch 啟用最快的 FlashAttention 引擎
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False) # 關閉慢速的傳統數學運算

    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)
    """ prepare log file """
    logs = set_log(cfg.OUTPUT_DIR, opt.cfg_file, opt.log_name)

    """ set single-gpu """
    device = torch.device(opt.device)
    # 🌟 [新增] 讓 cuDNN 自動尋找最快的卷積演算法 (因為輸入尺寸固定 128x128)
    torch.backends.cudnn.benchmark = True
    # [修改] 1. 設定全域 Config
    HandwritingDataset.set_global_config(cfg)

    """ set dataset"""
    # [修改] 2. 移除路徑參數，改用 split
    train_dataset = HandwritingDataset(split=cfg.TRAIN.TYPE, use_latent=True)
    test_dataset = HandwritingDataset(split=cfg.TEST.TYPE, use_latent=True)
    print('number of training images: ', len(train_dataset))
    # === 修改後的程式碼 ===
    # 移除 train_sampler 和 test_sampler
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                               drop_last=True,           
                                               collate_fn=train_dataset.collate_fn_,
                                               num_workers=cfg.DATA_LOADER.NUM_THREADS,
                                               pin_memory=True,          
                                               shuffle=True,             # 🌟 改為 shuffle=True
                                               prefetch_factor=8,        
                                               persistent_workers=False) 

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.TEST.IMS_PER_BATCH,
                                              drop_last=False,
                                              collate_fn=test_dataset.collate_fn_,
                                              pin_memory=True,
                                              num_workers=cfg.DATA_LOADER.NUM_THREADS,
                                              shuffle=False)             # 🌟 測試集不需要 shuffle

    # ----- 模型定義 -----
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, 
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
                     attention_resolutions=cfg.MODEL.ATTENTION_RESOLUTIONS, channel_mult=cfg.MODEL.CHANNEL_MULT, num_heads=cfg.MODEL.NUM_HEADS, 
                     context_dim=cfg.MODEL.EMB_DIM,
                     ).to(device)
    # 確保 PyTorch 版本支援 compile
    if hasattr(torch, 'compile'):
        print("🚀 啟動 torch.compile 加速 U-Net!")
        unet = torch.compile(unet)

    # ----- Pretrained 模型載入 -----
    if len(opt.one_dm) > 0:
        unet.load_state_dict(torch.load(opt.one_dm, map_location=torch.device('cpu')))
        print('Loaded pretrained one_dm model from {}'.format(opt.one_dm))

    """load pretrained resnet18 model"""
    if len(opt.feat_model) > 0:
        checkpoint = torch.load(opt.feat_model, map_location=torch.device('cpu'))
        checkpoint['conv1.weight'] = checkpoint['conv1.weight'].mean(1).unsqueeze(1)
        miss, unexp = unet.mix_net.Feat_Encoder.load_state_dict(checkpoint, strict=False)
        assert len(unexp) <= 32, "Failed to load the pretrained model"
        print('Loaded pretrained resnet18 model from {}'.format(opt.feat_model))
        

    # 加入 fused=True
    optimizer = optim.AdamW(unet.parameters(), lr=cfg.SOLVER.BASE_LR, fused=True)

    # ---- Resume Checkpoint if given ----
    start_epoch = 0
    if opt.resume_ckpt:
        resume_data = torch.load(opt.resume_ckpt, map_location='cpu')
        if 'model_state_dict' in resume_data:
            unet.load_state_dict(resume_data['model_state_dict'])
            optimizer.load_state_dict(resume_data['optimizer_state_dict'])
            start_epoch = resume_data.get('epoch', 0) + 1
            print(f"✅ Resumed from full checkpoint: {opt.resume_ckpt}, starting at epoch {start_epoch}")
        else:
            unet.load_state_dict(resume_data)
            print(f"⚠️  Loaded old-style checkpoint (only model weights) from {opt.resume_ckpt}")

    """build criterion and optimizer"""
    criterion = dict(nce=SupConLoss(contrast_mode='all'), recon=nn.MSELoss())
    diffusion = Diffusion(device=device, noise_offset=opt.noise_offset)

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    """Freeze vae and text_encoder"""
    vae.requires_grad_(False)
    vae = vae.to(device)

    """build trainer"""
    trainer = Trainer(diffusion, unet, vae, criterion, optimizer, train_loader, logs, test_loader, device)
    
    # 新增 start_epoch 給 Trainer 讓他知道從哪裡開始
    trainer.train(start_epoch=start_epoch)

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5', help='path to stable diffusion')
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64_scratch.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--feat_model', dest='feat_model', default='', help='pre-trained resnet18 model')
    parser.add_argument('--one_dm', dest='one_dm', default='', help='pre-trained one_dm model')
    parser.add_argument('--log', default='debug',
                        dest='log_name', required=False, help='the filename of log')
    parser.add_argument('--noise_offset', default=0, type=float, help='control the strength of noise')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--local_rank', type=int, default=0, help='device for training')
    parser.add_argument('--resume_ckpt', type=str, default='', help='Path to resume checkpoint (.pt)')
    opt = parser.parse_args()
    main(opt)