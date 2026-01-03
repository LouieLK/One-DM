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
                     context_dim=cfg.MODEL.EMB_DIM).to(device)

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

    # """load pretrained resnet18 model (Custom Mapping for backbone.x -> conv1/layer1...)"""
    # if len(opt.feat_model) > 0:
    #     print(f'Loading pretrained style encoder from {opt.feat_model} ...')
    #     checkpoint = torch.load(opt.feat_model, map_location=torch.device('cpu'))
        
    #     # 處理可能的 state_dict 包裝
    #     if 'state_dict' in checkpoint:
    #         checkpoint = checkpoint['state_dict']

    #     # 定義翻譯字典：將 Sequential 的索引映射回 ResNet 的標準名稱
    #     # 根據您的 inspect 結果：
    #     # backbone.0 -> conv1
    #     # backbone.1 -> bn1
    #     # backbone.4 -> layer1
    #     # backbone.5 -> layer2
    #     # backbone.6 -> layer3
    #     # backbone.7 -> layer4 (雖然 One-DM 不用 layer4，但載入也無妨)
    #     name_mapping = {
    #         'backbone.0.': 'conv1.',
    #         'backbone.1.': 'bn1.',
    #         'backbone.4.': 'layer1.',
    #         'backbone.5.': 'layer2.',
    #         'backbone.6.': 'layer3.',
    #         'backbone.7.': 'layer4.'
    #     }

    #     new_state_dict = {}
    #     for k, v in checkpoint.items():
    #         new_key = k
    #         # 1. 移除 'module.' (如果是 DDP 訓練的)
    #         if new_key.startswith('module.'):
    #             new_key = new_key[7:]
            
    #         # 2. 執行翻譯
    #         for old_prefix, new_prefix in name_mapping.items():
    #             if new_key.startswith(old_prefix):
    #                 new_key = new_key.replace(old_prefix, new_prefix, 1)
    #                 break # 找到對應前綴就停止
            
    #         # 3. 過濾掉不需要的層 (例如 proj. 投影層)
    #         # One-DM 的 ResNet 沒有 'proj' 或 'fc'
    #         if new_key.startswith('proj.') or new_key.startswith('fc.'):
    #             continue
                
    #         new_state_dict[new_key] = v

    #     checkpoint = new_state_dict

    #     # 再次檢查 key 是否正確
    #     if 'conv1.weight' not in checkpoint:
    #         print(f"⚠️ Warning: Mapping failed? Keys found: {list(checkpoint.keys())[:5]}")
    #     else:
    #         print(f"✅ Successfully mapped keys (e.g., backbone.0 -> conv1)")

    #     # 載入模型
    #     # strict=False 會自動忽略 layer4 (如果 One-DM 不需要) 以及 proj 層
    #     miss, unexp = unet.mix_net.Feat_Encoder.load_state_dict(checkpoint, strict=False)
    #     print(f"Loaded Feat_Encoder. Missing keys: {len(miss)}, Unexpected keys: {len(unexp)}")
        
    #     miss_f, unexp_f = unet.mix_net.freq_encoder.load_state_dict(checkpoint, strict=False)
    #     print(f"Loaded freq_encoder. Missing keys: {len(miss_f)}, Unexpected keys: {len(unexp_f)}")

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

    unet = DDP(unet, device_ids=[local_rank])
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