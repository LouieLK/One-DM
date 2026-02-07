import argparse
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.util import fix_seed, load_specific_dict
from utils.logger import set_log
from data_loader.loader import HandwritingDataset 
import torch
from trainer.trainer import Trainer
from models.unet import UNetModel
from torch import optim
import torch.nn as nn
from models.diffusion import Diffusion, EMA
import copy
from diffusers import AutoencoderKL
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models.loss import SupConLoss


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
    
    # [修改] 1. 設定全域 Config
    HandwritingDataset.set_global_config(cfg)

    """ set dataset"""
    # [修改] 2. 移除路徑參數，改用 split
    train_dataset = HandwritingDataset(split=cfg.TRAIN.TYPE)
    
    print('number of training images: ', len(train_dataset))
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                               drop_last=False,
                                               collate_fn=train_dataset.collate_fn_,
                                               num_workers=cfg.DATA_LOADER.NUM_THREADS,
                                               pin_memory=True,
                                               # [新增] 預取更多 batch，讓 CPU 跑在 GPU 前面
                                               prefetch_factor=4, 
                                               persistent_workers=True, # [新增] 避免每個 Epoch 重啟 worker
                                               sampler=train_sampler)
    
    
    # [修改] 3. 測試集同理
    test_dataset = HandwritingDataset(split=cfg.TEST.TYPE)
    test_sampler = DistributedSampler(test_dataset)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.TEST.IMS_PER_BATCH,
                                              drop_last=False,
                                              collate_fn=test_dataset.collate_fn_,
                                              pin_memory=True,
                                              num_workers=cfg.DATA_LOADER.NUM_THREADS,
                                              sampler=test_sampler)

    # ----- 模型定義 -----
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, 
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
                     attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS, 
                     context_dim=cfg.MODEL.EMB_DIM,backbone_type='mamba').to(device)

    # ----- Pretrained 模型載入 -----
    if len(opt.one_dm) > 0:
        unet.load_state_dict(torch.load(opt.one_dm, map_location=torch.device('cpu')))
        print('Loaded pretrained one_dm model from {}'.format(opt.one_dm))

    """load pretrained resnet18 model"""
    if len(opt.feat_model) > 0:
        print(f'🚀 Loading Style Encoder from: {opt.feat_model}')
        checkpoint = torch.load(opt.feat_model, map_location=torch.device('cpu'))
        
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        new_state_dict = {}
        for k, v in checkpoint.items():
            if 'fc' in k: continue # 跳過 fc
            
            new_k = k
            if new_k.startswith('module.'): new_k = new_k[7:]
            
            # === 關鍵修正：映射到 residual_function ===
            # 標準 ResNet -> ResNet Dilation
            # layer1 -> conv2_x
            # layer2 -> conv3_x
            # layer3 -> conv4_x
            # layer4 -> conv5_x
            
            if 'layer1' in new_k: 
                new_k = new_k.replace('layer1', 'conv2_x')
            elif 'layer2' in new_k: 
                new_k = new_k.replace('layer2', 'conv3_x')
            elif 'layer3' in new_k: 
                new_k = new_k.replace('layer3', 'conv4_x')
            elif 'layer4' in new_k: 
                new_k = new_k.replace('layer4', 'conv5_x')

            # 處理 BasicBlock 內部的映射
            # resnet_dilation 的 BasicBlock 用的是 residual_function Sequential
            # conv1 -> residual_function.0
            # bn1   -> residual_function.1
            # conv2 -> residual_function.3
            # bn2   -> residual_function.4
            # downsample -> shortcut
            
            parts = new_k.split('.')
            # parts 範例: ['conv2_x', '0', 'conv1', 'weight']
            
            if len(parts) >= 3 and parts[0].startswith('conv'):
                block_idx = parts[1] # '0'
                layer_name = parts[2] # 'conv1'
                
                prefix = f"{parts[0]}.{block_idx}"
                suffix = ".".join(parts[3:]) if len(parts) > 3 else ""
                
                if layer_name == 'conv1':
                    new_k = f"{prefix}.residual_function.0.{suffix}" if suffix else f"{prefix}.residual_function.0"
                elif layer_name == 'bn1':
                    new_k = f"{prefix}.residual_function.1.{suffix}" if suffix else f"{prefix}.residual_function.1"
                elif layer_name == 'conv2':
                    new_k = f"{prefix}.residual_function.3.{suffix}" if suffix else f"{prefix}.residual_function.3"
                elif layer_name == 'bn2':
                    new_k = f"{prefix}.residual_function.4.{suffix}" if suffix else f"{prefix}.residual_function.4"
                elif layer_name == 'downsample':
                    # downsample.0 -> shortcut.0
                    new_k = new_k.replace('downsample', 'shortcut')

            # 第一層
            if 'conv1.weight' in new_k: new_k = 'conv1.0.weight'
            if 'bn1.' in new_k: new_k = new_k.replace('bn1.', 'conv1.1.')

            new_state_dict[new_k] = v

        # 處理第一層 Conv (RGB -> Grayscale)
        if 'conv1.0.weight' in new_state_dict:
            w = new_state_dict['conv1.0.weight']
            if w.shape[1] == 3:
                print("   -> Converting weights from RGB (3ch) to Grayscale (1ch)")
                new_state_dict['conv1.0.weight'] = w.mean(dim=1, keepdim=True)

        miss, unexp = unet.mix_net.Feat_Encoder.load_state_dict(new_state_dict, strict=False)
        
        # 顯示真正重要的缺失層 (忽略 fc, avgpool)
        real_miss = [k for k in miss if 'residual_function' in k or 'conv1' in k]
        if len(real_miss) > 0:
            print(f"⚠️ Warning: Still missing layers: {real_miss[:5]}")
        else:
            print(f"✅ Successfully loaded fine-tuned weights into Feat_Encoder!")

    optimizer = optim.AdamW(unet.parameters(), lr=cfg.SOLVER.BASE_LR)

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

    unet = DDP(unet, device_ids=[local_rank],find_unused_parameters=True)

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
    parser.add_argument('--log_name', default='debug',
                        dest='log_name', required=False, help='the filename of log')
    parser.add_argument('--noise_offset', default=0, type=float, help='control the strength of noise')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--local_rank', type=int, default=0, help='device for training')
    parser.add_argument('--resume_ckpt', type=str, default='', help='Path to resume checkpoint (.pt)')
    opt = parser.parse_args()
    main(opt)