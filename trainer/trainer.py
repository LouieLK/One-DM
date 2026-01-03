import torch
from tensorboardX import SummaryWriter
import time
from parse_config import cfg
import os
import sys
from PIL import Image
import torchvision
from tqdm import tqdm
from data_loader.loader import ContentData
import torch.distributed as dist
import torch.nn.functional as F
import random

class Trainer:
    def __init__(self, diffusion, unet, vae, criterion, optimizer, data_loader, 
                logs, valid_data_loader=None, device=None, ocr_model=None, ctc_loss=None):
        self.model = unet
        self.diffusion = diffusion
        self.vae = vae
        self.recon_criterion = criterion['recon']
        self.nce_criterion = criterion['nce']
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.tb_summary = SummaryWriter(logs['tboard'])
        self.save_model_dir = logs['model']
        self.save_sample_dir = logs['sample']
        self.ocr_model = ocr_model
        self.ctc_criterion = ctc_loss
        self.device = device
      
    def _train_iter(self, data, step, pbar):
        self.model.train()
        # prepare input

        images, style_ref, laplace_ref, content_ref, wid = data['img'].to(self.device), \
            data['style'].to(self.device), \
            data['laplace'].to(self.device), \
            data['content'].to(self.device), \
            data['wid'].to(self.device)
        
        # vae encode
        images = self.vae.encode(images).latent_dist.sample()
        images = images * 0.18215


        # forward
        t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(images, t)
        
       
        predicted_noise, high_nce_emb, low_nce_emb = self.model(x_t, t, style_ref, laplace_ref, content_ref, tag='train')
        # calculate loss
        recon_loss = self.recon_criterion(predicted_noise, noise)
        
        high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
        low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)
        
        loss = recon_loss + (high_nce_loss * 1.0) + (low_nce_loss * 1.0)
        # backward and update trainable parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if dist.get_rank() == 0:
            # log file
            loss_dict = {"reconstruct_loss": recon_loss.item(), "high_nce_loss": high_nce_loss.item(),
                         "low_nce_loss": low_nce_loss.item()}
            self.tb_summary.add_scalars("loss", loss_dict, step)
            self._progress(recon_loss.item(), pbar)

        del data, loss
        torch.cuda.empty_cache()

    def _finetune_iter(self, data, step, pbar):
        self.model.train()
        # prepare input

        images, style_ref, laplace_ref, content_ref, wid, target, target_lengths = data['img'].to(self.device), \
            data['style'].to(self.device), \
            data['laplace'].to(self.device), \
            data['content'].to(self.device), \
            data['wid'].to(self.device), \
            data['target'].to(self.device), \
            data['target_lengths'].to(self.device)
        
        # vae encode
        latent_images = self.vae.encode(images).latent_dist.sample()
        latent_images = latent_images * 0.18215


        # forward
        t = self.diffusion.sample_timesteps(latent_images.shape[0], finetune=True).to(self.device)
        x_t, noise = self.diffusion.noise_images(latent_images, t)
        
        x_start, predicted_noise, high_nce_emb, low_nce_emb = self.diffusion.train_ddim(self.model, x_t, style_ref, laplace_ref,
                                                        content_ref, t, sampling_timesteps=5)
 
        # calculate loss
        recon_loss = self.recon_criterion(predicted_noise, noise)
        rec_out = self.ocr_model(x_start)
        input_lengths = torch.IntTensor(x_start.shape[0]*[rec_out.shape[0]])
        ctc_loss = self.ctc_criterion(F.log_softmax(rec_out, dim=2), target, input_lengths, target_lengths)
        high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
        low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)
        loss = recon_loss + high_nce_loss + low_nce_loss + 0.1*ctc_loss

        # backward and update trainable parameters
        self.optimizer.zero_grad()
        loss.backward()
        if cfg.SOLVER.GRAD_L2_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.SOLVER.GRAD_L2_CLIP)
        self.optimizer.step()

        if dist.get_rank() == 0:
            # log file
            loss_dict = {"reconstruct_loss": recon_loss.item(), "high_nce_loss": high_nce_loss.item(),
                         "low_nce_loss": low_nce_loss.item(), "ctc_loss": ctc_loss.item()}
            self.tb_summary.add_scalars("loss", loss_dict, step)
            self._progress(recon_loss.item(), pbar)

        del data, loss
        torch.cuda.empty_cache()

    def _save_images(self, images, path):
        grid = torchvision.utils.make_grid(images)
        im = torchvision.transforms.ToPILImage()(grid)
        im.save(path)
        return im

    # @torch.no_grad()
    # def _valid_iter(self, epoch):
    #     print('loading test dataset, the number is', len(self.valid_data_loader))
    #     self.model.eval()
    #     # use the first batch of dataloader in all validations for better visualization comparisons
    #     test_loader_iter = iter(self.valid_data_loader)
    #     test_data = next(test_loader_iter)
    #     # prepare input
    #     images, style_ref, laplace_ref, content_ref = test_data['img'].to(self.device), \
    #         test_data['style'].to(self.device), \
    #         test_data['laplace'].to(self.device), \
    #         test_data['content'].to(self.device)
    
    #     load_content = ContentData()
    #     # forward
    #     texts = ['getting', 'both', 'success']
    #     for text in texts:
    #         rank = dist.get_rank()
    #         text_ref = load_content.get_content(text)
    #         text_ref = text_ref.to(self.device).repeat(style_ref.shape[0], 1, 1, 1)
    #         x = torch.randn((text_ref.shape[0], 4, style_ref.shape[2]//8, (text_ref.shape[1]*32)//8)).to(self.device)
    #         preds = self.diffusion.ddim_sample(self.model, self.vae, images.shape[0], x, style_ref, laplace_ref, text_ref)
    #         out_path = os.path.join(self.save_sample_dir, f"epoch-{epoch}-{text}-process-{rank}.png")
    #         self._save_images(preds, out_path)

    # @torch.no_grad()
    # def _valid_iter(self, epoch):
    #     print('loading test dataset, the number is', len(self.valid_data_loader))
    #     self.model.eval()
        
    #     test_loader_iter = iter(self.valid_data_loader)
    #     test_data = next(test_loader_iter)
        
    #     # 準備資料
    #     images, style_ref_pair, laplace_ref_pair, content_ref = test_data['img'].to(self.device), \
    #         test_data['style'].to(self.device), \
    #         test_data['laplace'].to(self.device), \
    #         test_data['content'].to(self.device)
    
    #     # 1. 選取 View 1 作為風格參考
    #     style_ref = style_ref_pair[:, 1:2]     # Shape: [Batch, 1, 64, 64] (CUDA)
    #     laplace_ref = laplace_ref_pair[:, 1:2] # Shape: [Batch, 1, 64, 64] (CUDA)

    #     # 2. 準備隨機文字
    #     load_content = ContentData()
    #     if hasattr(load_content, 'letters') and len(load_content.letters) >= 5:
    #         selected_texts = random.sample(load_content.letters, 5)
            
    #     print(f"Validation Generating Texts: {selected_texts}")

    #     for text in selected_texts:
    #         rank = dist.get_rank()
    #         try:
    #             text_ref = load_content.get_content(text)
    #         except KeyError as e:
    #             print(f"Warning: Character {text} not in dictionary, skipping...")
    #             continue

    #         text_ref = text_ref.to(self.device).repeat(style_ref.shape[0], 1, 1, 1)
            
    #         # 設定寬度為 64 (符合中文字)
    #         x = torch.randn((text_ref.shape[0], 4, style_ref.shape[2]//8, (text_ref.shape[1]*64)//8)).to(self.device)
            
    #         # 3. 執行生成 (preds 會回傳 CPU Tensor, 數值範圍 0~1)
    #         preds = self.diffusion.ddim_sample(self.model, self.vae, images.shape[0], x, style_ref, laplace_ref, text_ref)
            
    #         # 4. [修正] 處理 style_ref 以便拼接
    #         #   a. 搬移到 CPU (因為 preds 在 CPU)
    #         #   b. 反正規化: -1~1 -> 0~1 (因為 preds 已經是 0~1)
    #         style_ref_cpu = style_ref.cpu()
    #         style_ref_cpu = (style_ref_cpu * 0.5 + 0.5).clamp(0, 1)
    #         if style_ref_cpu.shape[1] == 1:
    #             style_ref_cpu = style_ref_cpu.repeat(1, 3, 1, 1)
    #         # 5. 製作對照圖：[參考圖 | 生成圖]
    #         # 現在兩者都在 CPU 且都是 0~1，可以直接拼接
    #         comparison = torch.cat([style_ref_cpu, preds], dim=3)
            
    #         # 6. 儲存 (不需要再做反正規化了)
    #         out_path = os.path.join(self.save_sample_dir, f"epoch-{epoch}-{text}-process-{rank}.png")
    #         self._save_images(comparison, out_path)

    @torch.no_grad()
    def _valid_iter(self, epoch):
        print('loading test dataset, the number is', len(self.valid_data_loader))
        self.model.eval()
        
        # 1. 取得一個 Batch 的測試資料
        test_loader_iter = iter(self.valid_data_loader)
        test_data = next(test_loader_iter)
        
        # 準備資料
        # 注意: 這裡假設 loader 讀進來的 style 已經是 [B, 2, 64, 64] (因為有 View 1 & View 2)
        images = test_data['img'].to(self.device)
        style_ref_pair = test_data['style'].to(self.device)
        laplace_ref_pair = test_data['laplace'].to(self.device)
        
        # 取出第一張 View 作為參考圖 [Batch, 1, 64, 64]
        style_ref = style_ref_pair[:, 0:1, :, :] 
        laplace_ref = laplace_ref_pair[:, 0:1, :, :]

        # 2. 準備要生成的文字列表
        load_content = ContentData()
        # 這裡可以自訂想要測試的字，或者隨機選取
        if hasattr(load_content, 'letters') and len(load_content.letters) >= 5:
            # 隨機選 5 個字來測試
            selected_texts = random.sample(load_content.letters, 5) 
        else:
            # 如果讀不到 letters，就用預設的
            selected_texts = ['永', '和', '九', '年', '歲']

        print(f"Validation Generating Texts: {selected_texts}")

        for text in selected_texts:
            rank = dist.get_rank()
            
            # 取得 Content Reference (字形內容)
            try:
                text_ref = load_content.get_content(text)
            except KeyError:
                print(f"Warning: Character {text} not in dictionary, skipping...")
                continue

            # 複製 Content Ref 以符合 Batch Size
            text_ref = text_ref.to(self.device).repeat(style_ref.shape[0], 1, 1, 1)
            
            # 3. 建立初始雜訊 x_T
            # 因為資料集固定為 64x64，且 U-Net 下採樣 8 倍 (2^3)，所以 Latent Size 為 8x8
            h_latent = 64 // 8
            w_latent = 64 // 8
            x = torch.randn((text_ref.shape[0], 4, h_latent, w_latent)).to(self.device)
            
            # 4. 執行生成 (DDIM Sampling)
            # preds 數值範圍通常為 0~1 (視 model 輸出而定)
            preds = self.diffusion.ddim_sample(self.model, self.vae, images.shape[0], x, style_ref, laplace_ref, text_ref)
            
            # 5. 製作對照圖 (Style Ref | Generated Image)
            
            # 處理 Style Image (搬到 CPU, 反正規化)
            style_vis = style_ref.cpu()
            # 假設輸入時有做 Normalize(0.5, 0.5)，這裡還原回 0~1
            style_vis = (style_vis * 0.5 + 0.5).clamp(0, 1)
            
            # [關鍵修正] 確保通道數一致
            # 如果生成圖是 RGB (3通道)，但風格圖是灰階 (1通道)，將風格圖複製成 3 通道
            if preds.shape[1] == 3 and style_vis.shape[1] == 1:
                style_vis = style_vis.repeat(1, 3, 1, 1)
            
            # 確保高度一致 (理論上都是 64，但為了安全起見)
            if style_vis.shape[2] != preds.shape[2] or style_vis.shape[3] != preds.shape[3]:
                 style_vis = torch.nn.functional.interpolate(style_vis, size=(preds.shape[2], preds.shape[3]), mode='bilinear')

            # 左右拼接 (dim=3 是寬度方向)
            comparison = torch.cat([style_vis, preds.cpu()], dim=3)
            
            # 6. 存檔
            out_path = os.path.join(self.save_sample_dir, f"epoch-{epoch}-{text}-process-{rank}.png")
            self._save_images(comparison, out_path)
            # print(f"Saved validation image to {out_path}")

    def train(self, start_epoch=0):
        """start training iterations"""
        for epoch in range(start_epoch,cfg.SOLVER.EPOCHS):
            self.data_loader.sampler.set_epoch(epoch)
            print(f"Epoch:{epoch} of process {dist.get_rank()}")
            dist.barrier()
            if dist.get_rank() == 0:
                pbar = tqdm(self.data_loader, leave=False)
            else:
                pbar = self.data_loader

            for step, data in enumerate(pbar):
                total_step = epoch * len(self.data_loader) + step
                if self.ocr_model is not None:
                    self._finetune_iter(data, total_step, pbar)
                    if (total_step+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (total_step+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                        if dist.get_rank() == 0:
                            self._save_checkpoint(total_step)
                    else:
                        pass
                    if self.valid_data_loader is not None:
                        if (total_step+1) > cfg.TRAIN.VALIDATE_BEGIN  and (total_step+1) % cfg.TRAIN.VALIDATE_ITERS == 0:
                            self._valid_iter(total_step)
                        else:
                            pass 
                else:
                    self._train_iter(data, total_step, pbar)

            if (epoch+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (epoch+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                if dist.get_rank() == 0:
                    self._save_checkpoint(epoch)
                else:
                    pass
            if self.valid_data_loader is not None:
                if (epoch+1) > cfg.TRAIN.VALIDATE_BEGIN  and (epoch+1) % cfg.TRAIN.VALIDATE_ITERS == 0:
                    self._valid_iter(epoch)
            else:
                pass

            if dist.get_rank() == 0:
                pbar.close()

    def _progress(self, loss, pbar):
        pbar.set_postfix(mse='%.6f' % (loss))

    def _save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        ckpt_path = os.path.join(self.save_model_dir, f'{epoch}-ckpt.pt')
        torch.save(checkpoint, ckpt_path)
        print(f"✅ Saved checkpoint to {ckpt_path}")