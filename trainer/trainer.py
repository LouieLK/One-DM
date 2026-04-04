import torch
from tensorboardX import SummaryWriter
from parse_config import cfg
import os
import torchvision
from tqdm import tqdm
from data_loader.loader import ContentData
import torch.nn.functional as F
import random
from torch.amp import autocast
import gc # 放在檔案最開頭
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
        drop_style = random.random() < 0.1     # 10% 機率丟棄風格
        drop_content = random.random() < 0.1   # 10% 機率丟棄內容
        is_style_uncond = False # 標記旗標

        if drop_style:
            style_ref = torch.zeros_like(style_ref)
            laplace_ref = torch.zeros_like(laplace_ref)
            is_style_uncond = True 

        if drop_content:
            content_ref = torch.zeros_like(content_ref)

        # vae encode
        if images.shape[1] == 3:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    images = self.vae.encode(images).latent_dist.sample()
                    images = images * 0.18215

        # forward
        t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(images, t)

        # x_t_double = torch.cat([x_t, x_t], dim=0)
        # t_double = torch.cat([t, t], dim=0)
        # style_double = torch.cat([style_ref, style_ref], dim=0)
        # laplace_double = torch.cat([laplace_ref, laplace_ref], dim=0)
        # content_double = torch.cat([content_ref, content_masked], dim=0)

        # with autocast(device_type='cuda', dtype=torch.bfloat16):
        #     # ==========================================
        #     # 2. 只做「一次」前向傳播！徹底杜絕內部 Inplace 覆蓋
        #     # ==========================================
        #     preds_double, high_nce_double, low_nce_double = self.model(
        #         x_t_double, t_double, style_double, laplace_double, content_double, tag='train'
        #     )

        #     # ==========================================
        #     # 3. 將輸出的結果切回兩半 (前半是 ref，後半是 masked)
        #     # ==========================================
        #     predicted_noise, predicted_noise_masked = preds_double.chunk(2, dim=0)
            
        #     # NCE embedding 我們只需要算前半段 (ref) 的 loss，後半段丟棄即可
        #     high_nce_emb, _ = high_nce_double.chunk(2, dim=0) 
        #     low_nce_emb, _ = low_nce_double.chunk(2, dim=0)
        #     # ==========================================
        #     # 4. 計算 Loss
        #     # ==========================================
        #     recon_loss_full = self.recon_criterion(predicted_noise, noise)
        #     recon_loss_masked = self.recon_criterion(predicted_noise_masked, noise)

        #     loss_consistency = F.l1_loss(
        #         predicted_noise_masked,
        #         predicted_noise.detach()
        #     )

        #     if is_style_uncond:
        #         high_nce_loss = torch.tensor(0.0, device=self.device)
        #         low_nce_loss = torch.tensor(0.0, device=self.device)
        #     else:
        #         # 只有在有風格輸入時，才計算風格損失
        #         high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
        #         low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)

        #     loss = (
        #         recon_loss_full
        #         + 1.0 * recon_loss_masked
        #         + 1.0 * high_nce_loss
        #         + 1.0 * low_nce_loss
        #         + 1.0 * loss_consistency
        #     )
        # backward and update trainable parameters
        # 找到 self.optimizer.zero_grad() 並替換為：

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # 正常送入單一 Batch
            predicted_noise, high_nce_emb, low_nce_emb = self.model(
                x_t, t, style_ref, laplace_ref, content_ref, tag='train'
            )
            # ==========================================
            # 計算 Loss (移除 consistency_loss)
            # ==========================================
            recon_loss = self.recon_criterion(predicted_noise, noise)
            
            if is_style_uncond:
                high_nce_loss = torch.tensor(0.0, device=self.device)
                low_nce_loss = torch.tensor(0.0, device=self.device)
            else:
                high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
                low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)
            
            # [修改] 總 Loss 拔除 consistency_loss
            loss = recon_loss + (high_nce_loss * 1.0) + (low_nce_loss * 1.0)
            
        # backward
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # log file
        loss_dict = {"reconstruct_loss": recon_loss.item(), "high_nce_loss": high_nce_loss.item(),
                        "low_nce_loss": low_nce_loss.item()} # 移除 consistency_loss 紀錄
        self.tb_summary.add_scalars("loss", loss_dict, step)
        self._progress(recon_loss.item(), pbar)

        del data, loss

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
        
        # ===== [修改] Classifier-Free Guidance 雙重隨機 Dropout =====
        drop_style = random.random() < 0.1     # 10% 機率丟棄風格
        drop_content = random.random() < 0.1   # 10% 機率丟棄內容
        is_style_uncond = False # 標記旗標
        is_content_uncond = False  # [新增] 
        if drop_style:
            style_ref = torch.zeros_like(style_ref)
            laplace_ref = torch.zeros_like(laplace_ref)
            is_style_uncond = True 

        if drop_content:
            content_ref = torch.zeros_like(content_ref)
            is_content_uncond = True   # [新增]
        # vae encode
        if images.shape[1] == 3:
            with torch.no_grad(): # 凍結的 VAE 不需要算梯度
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    latent_images = self.vae.encode(images).latent_dist.sample()
                    latent_images = latent_images * 0.18215
        else:
            # 如果已經是從硬碟讀出來的 Latent (4 通道)，直接沿用
            latent_images = images


        # forward
        t = self.diffusion.sample_timesteps(latent_images.shape[0], finetune=True).to(self.device)
        x_t, noise = self.diffusion.noise_images(latent_images, t)
        
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # 取得預測結果
            # train_ddim 內部應該已經有計算 x_start 的邏輯
            x_start, predicted_noise, high_nce_emb, low_nce_emb = self.diffusion.train_ddim(
                self.model, x_t, style_ref, laplace_ref, content_ref, t, sampling_timesteps=5
            )
            
            # calculate loss
            recon_loss = self.recon_criterion(predicted_noise, noise)

            if is_style_uncond:
                high_nce_loss = torch.tensor(0.0, device=self.device)
                low_nce_loss = torch.tensor(0.0, device=self.device)
            else:
                # 只有在有風格輸入時，才計算風格損失
                high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
                low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)

            # [修改] 如果沒有 Content，就絕對不能算 CTC Loss！
            if is_content_uncond:
                ctc_loss = torch.tensor(0.0, device=self.device)
            else:
                rec_out = self.ocr_model(x_start)
                input_lengths = torch.IntTensor(x_start.shape[0]*[rec_out.shape[0]])
                target_shifted = target + 1 
                ctc_loss = self.ctc_criterion(F.log_softmax(rec_out, dim=2), target_shifted, input_lengths, target_lengths)

            # 總 Loss
            loss = recon_loss + high_nce_loss + low_nce_loss + 0.1 * ctc_loss

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        if cfg.SOLVER.GRAD_L2_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.SOLVER.GRAD_L2_CLIP)
        self.optimizer.step()

        loss_dict = {
            "reconstruct_loss": recon_loss.item(), 
            "high_nce_loss": high_nce_loss.item(),
            "low_nce_loss": low_nce_loss.item(), 
            "ctc_loss": ctc_loss.item()
        }
        self.tb_summary.add_scalars("loss", loss_dict, step)
        self._progress(recon_loss.item(), pbar)

        del data, loss

    def _save_images(self, images, path):
        grid = torchvision.utils.make_grid(images)
        im = torchvision.transforms.ToPILImage()(grid)
        im.save(path)
        return im



    # [新增] 用來計算驗證集上的 Loss (不進行生成，只算數學指標)
    @torch.no_grad()
    def _validate_loss(self, epoch):
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_ctc = 0
        count = 0
        
        # 顯示進度條
        pbar = tqdm(self.valid_data_loader, desc=f"Validating Epoch {epoch}", leave=False)
        
        for data in pbar:
            # 1. 準備資料
            # 注意：這裡的 key 必須跟 loader 輸出的 dict 一致
            images = data['img'].to(self.device)
            style_ref = data['style'].to(self.device)
            laplace_ref = data['laplace'].to(self.device)
            content_ref = data['content'].to(self.device)
            wid = data['wid'].to(self.device)
            target = data['target'].to(self.device)
            target_lengths = data['target_lengths'].to(self.device)

            # 2. VAE Encode
            gt_latents = self.vae.encode(images).latent_dist.sample().mul_(0.18215)
            
            # 3. 加噪
            t = self.diffusion.sample_timesteps(gt_latents.shape[0]).to(self.device)
            x_t, noise = self.diffusion.noise_images(gt_latents, t)

            # 4. UNet 預測
            predicted_noise, high_nce_emb, low_nce_emb = self.model(x_t, t, style_ref, laplace_ref, content_ref, tag='train')
            
            # 5. 計算 Recon Loss
            loss_recon = self.recon_criterion(noise, predicted_noise)
            
            # 6. 計算 NCE Loss (Style)
            if wid is not None:
                high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
                low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)
            else:
                high_nce_loss = 0
                low_nce_loss = 0
            
            # 7. 計算 CTC Loss (關鍵指標：字有沒有寫對)
            loss_ctc = torch.tensor(0.0, device=self.device)
            
            # 加入 Time Masking 邏輯 (同訓練)
            ocr_limit = 400
            mask_t = (t < ocr_limit).float().view(-1, 1, 1, 1)
            
            if mask_t.sum() > 0:
                alpha_hat = self.diffusion.alpha_hat[t][:, None, None, None]
                denom = torch.sqrt(alpha_hat).clamp(min=1e-5)
                pred_z0 = (x_t - torch.sqrt(1 - alpha_hat) * predicted_noise) / denom
                pred_z0 = pred_z0.clamp(-5, 5)
                masked_z0 = pred_z0 * mask_t
                
                # OCR Forward
                ocr_preds = self.ocr_model(masked_z0)
                if ocr_preds.size(0) == gt_latents.size(0):
                    ocr_preds = ocr_preds.permute(1, 0, 2)
                
                T_seq, B_seq, _ = ocr_preds.shape
                ocr_log_probs = ocr_preds.log_softmax(2)
                input_lengths = torch.full(size=(B_seq,), fill_value=T_seq, dtype=torch.long).to(self.device)
                
                # 解決 Index 衝突
                target_shifted = target + 1
                
                loss_ctc_raw = F.ctc_loss(ocr_log_probs, target_shifted, input_lengths, target_lengths, 
                                          blank=0, zero_infinity=True, reduction='none')
                
                valid_loss = loss_ctc_raw * mask_t.view(-1)
                loss_ctc = valid_loss.sum() / mask_t.sum()

            # 8. 總結 Loss
            # 這裡使用跟訓練一樣的比例，例如 0.01 * ctc
            current_loss = loss_recon + high_nce_loss + low_nce_loss + 0.01 * loss_ctc
            
            total_loss += current_loss.item()
            total_recon += loss_recon.item()
            total_ctc += loss_ctc.item()
            count += 1
            
        avg_loss = total_loss / count
        avg_recon = total_recon / count
        avg_ctc = total_ctc / count
        
        # 寫入 Tensorboard
        val_dict = {
            "val_total_loss": avg_loss,
            "val_recon": avg_recon,
            "val_ctc": avg_ctc
        }
        self.tb_summary.add_scalars("validation", val_dict, epoch)
        print(f"\n[Validation] Epoch {epoch} | Total: {avg_loss:.4f} | Recon: {avg_recon:.4f} | CTC: {avg_ctc:.4f}")
            
        return avg_loss
    
    @torch.no_grad()
    def _valid_iter(self, epoch):
        print('loading test dataset, the number is', len(self.valid_data_loader))
        self.model.eval()
        
        # 1. 取得一個 Batch 的測試資料
        test_loader_iter = iter(self.valid_data_loader)
        test_data = next(test_loader_iter)
        
        # 準備資料
        # 注意: 這裡假設 loader 讀進來的 style 已經是 [B, 2, 128, 128] (因為有 View 1 & View 2)
        images = test_data['img'].to(self.device)
        style_ref_pair = test_data['style'].to(self.device)
        laplace_ref_pair = test_data['laplace'].to(self.device)
        
        # 取出第一張 View 作為參考圖 [Batch, 1, 128, 128]
        style_ref = style_ref_pair[:, 0:1, :, :] 
        laplace_ref = laplace_ref_pair[:, 0:1, :, :]

        # 2. 準備要生成的文字列表
        load_content = ContentData()
        # 這裡可以自訂想要測試的字，或者隨機選取
        unseen = '着紫仔自总'
        if hasattr(load_content, 'letters') and len(load_content.letters) >= 5:
            # 隨機選 5 個字來測試
            selected_texts = random.sample(load_content.letters, 3) 
        else:
            # 如果讀不到 letters，就用預設的
            selected_texts = ['永', '和', '九', '年', '歲']

        selected_texts.extend(unseen)
        print(f"Validation Generating Texts: {selected_texts}")

        for text in selected_texts:
            
            # 取得 Content Reference (字形內容)
            try:
                text_ref = load_content.get_content(text)
            except KeyError:
                print(f"Warning: Character {text} not in dictionary, skipping...")
                continue

            # 複製 Content Ref 以符合 Batch Size
            text_ref = text_ref.to(self.device).repeat(style_ref.shape[0], 1, 1, 1)
            
            # 3. 建立初始雜訊 x_T
            # [修改] 因為資料集固定為 128x128，且 U-Net 下採樣 8 倍 (2^3)，所以 Latent Size 改為 16x16
            h_latent = 128 // 8
            w_latent = 128 // 8
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
            
            # 確保高度一致 (理論上都是 128，但為了安全起見)
            if style_vis.shape[2] != preds.shape[2] or style_vis.shape[3] != preds.shape[3]:
                 style_vis = torch.nn.functional.interpolate(style_vis, size=(preds.shape[2], preds.shape[3]), mode='bilinear')

            # 左右拼接 (dim=3 是寬度方向)
            comparison = torch.cat([style_vis, preds.cpu()], dim=3)
            
            # 6. 存檔
            out_path = os.path.join(self.save_sample_dir, f"epoch-{epoch}-{text}.png")
            self._save_images(comparison, out_path)
            # print(f"Saved validation image to {out_path}")

    def train(self, start_epoch=0):
        best_val_loss = float('inf')
        """start training iterations"""
        for epoch in range(start_epoch,cfg.SOLVER.EPOCHS):
            print(f"Epoch:{epoch}")
            pbar = tqdm(self.data_loader, leave=False)
            for step, data in enumerate(pbar):
                total_step = epoch * len(self.data_loader) + step
                if self.ocr_model is not None:
                    self._finetune_iter(data, total_step, pbar)
                    if (total_step+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (total_step+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
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
                self._save_checkpoint(epoch)
            if self.valid_data_loader is not None:
                if (epoch+1) > cfg.TRAIN.VALIDATE_BEGIN  and (epoch+1) % cfg.TRAIN.VALIDATE_ITERS == 0:
                    self._valid_iter(epoch)
                # # 2. [新增] 計算 Loss (選模型用)
                # # 這裡假設每個 Epoch 都算一次 Loss
                # current_val_loss = self._validate_loss(epoch)
                
                # # 3. 自動儲存最佳模型
                # if dist.get_rank() == 0:
                #     if current_val_loss < best_val_loss:
                #         best_val_loss = current_val_loss
                #         best_ckpt_path = os.path.join(self.save_model_dir, 'best_model.pt')
                #         self._save_checkpoint(epoch) # 存這個 epoch 的檔
                #         # 另外複製一份作為 best
                #         torch.save({
                #             'epoch': epoch,
                #             'model_state_dict': self.model.module.state_dict(),
                #             'optimizer_state_dict': self.optimizer.state_dict(),
                #             'best_val_loss': best_val_loss
                #         }, best_ckpt_path)
                #         print(f"★ New Best Model Saved! Loss: {best_val_loss:.4f}")
            else:
                pass

            pbar.close()
            gc.collect() 

    def _progress(self, loss, pbar):
        pbar.set_postfix(mse='%.6f' % (loss))

    def _save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        ckpt_path = os.path.join(self.save_model_dir, f'{epoch}-ckpt.pt')
        torch.save(checkpoint, ckpt_path)
        print(f"✅ Saved checkpoint to {ckpt_path}")