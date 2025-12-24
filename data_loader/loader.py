import random
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pickle
from torchvision import transforms
from PIL import Image
import torchvision
import cv2
from data_loader.laplace import laplace_transform
class HandwritingDataset(Dataset):
    _global_cfg = None 

    @classmethod
    def set_global_config(cls, cfg):
        """設定全域 Config，所有實例化物件皆可共用"""
        cls._global_cfg = cfg

    def __init__(self, cfg=None, split='train', content_type='unifont'):
        # 1. Config 解析
        if cfg is not None:
            self.cfg = cfg
        elif self._global_cfg is not None:
            self.cfg = self._global_cfg
        else:
            raise ValueError("Configuration not set! Call HandwritingDataset.set_global_config(cfg) first.")

        ds_cfg = self.cfg['DATASET']
        self.root = ds_cfg['ROOT']
        self.letters = ds_cfg['LETTERS']
        self.max_len = ds_cfg['MAX_LEN']
        self.style_len = ds_cfg['STYLE_LEN']
        
        txt_path = os.path.join(self.root, ds_cfg['FILES'][split])
        self.data_dict = self.load_data(txt_path)
        # 2. 路徑設定
        self.image_path = os.path.join(self.root, ds_cfg['DIRS']['IMAGE'],split)
        self.style_path = os.path.join(self.root, ds_cfg['DIRS']['STYLE'],split)
        self.laplace_path = os.path.join(self.root, ds_cfg['DIRS']['LAPLACE'],split)

        self.tokens = {"PAD_TOKEN": len(self.letters)}
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.indices = list(self.data_dict.keys())
        self.transforms = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
        #self.content_transform = torchvision.transforms.Resize([64, 32], interpolation=Image.NEAREST)
        self.con_symbols = self.get_symbols(content_type)
        self.laplace = torch.tensor([[0, 1, 0],[1, -4, 1],[0, 1, 0]], dtype=torch.float
                                    ).to(torch.float32).view(1, 1, 3, 3).contiguous()



    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            train_data = f.readlines()
            train_data = [i.strip().split(' ') for i in train_data]
            full_dict = {}
            idx = 0
            for i in train_data:
                s_id = i[0].split(',')[0]
                image = i[0].split(',')[1] + '.png'
                transcription = i[1]
                if len(transcription) > self.max_len:
                    continue
                full_dict[idx] = {'image': image, 's_id': s_id, 'label':transcription}
                idx += 1
        return full_dict
    
    # prepare style-preserving augment for 64x64 handwritten chinese glyph
    # def _prepare_style_augment(self, img):

    #     # assume img is gray numpy array, uint8, white background (255), dark strokes (0)
    #     h, w = img.shape[:2]

    #     # upscale to operate on higher res and preserve stroke detail
    #     up = 256
    #     img_up = cv2.resize(img, (up, up), interpolation=cv2.INTER_LINEAR)

    #     # -------------------------
    #     # 1) conservative random crop / small shift
    #     # scale limited to 0.92 - 1.0 to avoid removing stroke parts
    #     s = random.uniform(0.92, 1.0)
    #     ch = int(up * s)
    #     cw = ch
    #     sy = random.randint(0, max(0, up - ch))
    #     sx = random.randint(0, max(0, up - cw))
    #     img_c = img_up[sy:sy+ch, sx:sx+cw]
    #     img_c = cv2.resize(img_c, (up, up), interpolation=cv2.INTER_LINEAR)

    #     # -------------------------
    #     # 2) affine: rotation, small trans, scale, and small shear (properly applied)
    #     # rotation range reduced (±6 deg), shear small (±6 deg)
    #     angle = random.uniform(-6.0, 6.0)
    #     tx = random.uniform(-2.0, 2.0)  # translation in px at up scale
    #     ty = random.uniform(-2.0, 2.0)
    #     zoom = random.uniform(0.97, 1.03)
    #     shear_deg = random.uniform(-6.0, 6.0)
    #     shear_rad = np.deg2rad(shear_deg)

    #     cx = up / 2.0
    #     cy = up / 2.0

    #     # rotation+scale matrix (2x3) -> convert to 3x3
    #     Mrs = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    #     Mrs_3 = np.vstack([Mrs, [0.0, 0.0, 1.0]])  # shape (3,3)

    #     # shear matrix (x-shear). small shear to avoid changing stroke geometry much.
    #     Sh = np.array([
    #         [1.0, np.tan(shear_rad), 0.0],
    #         [0.0, 1.0, 0.0],
    #         [0.0, 0.0, 1.0]
    #     ], dtype=np.float32)

    #     # translation matrix
    #     T = np.array([
    #         [1.0, 0.0, tx],
    #         [0.0, 1.0, ty],
    #         [0.0, 0.0, 1.0]
    #     ], dtype=np.float32)

    #     # combine: first apply rotation/scale, then shear, then translation
    #     M_comb = T @ Sh @ Mrs_3
    #     M2 = M_comb[0:2, :]  # cv2.warpAffine needs 2x3

    #     img_af = cv2.warpAffine(img_c, M2, (up, up),
    #                             flags=cv2.INTER_LINEAR,
    #                             borderMode=cv2.BORDER_CONSTANT,
    #                             borderValue=255)

    #     # -------------------------
    #     # 3) morphology: very small chance and mild kernel
    #     # Be explicit: erosion -> thin strokes; dilation -> thicken strokes
    #     # If style should preserve stroke width, consider setting p_morph = 0
    #     p_morph = 0.15
    #     if random.random() < p_morph:
    #         # use small kernel; iterations=1 only
    #         k = np.ones((2, 2), np.uint8)
    #         if random.random() < 0.5:
    #             # erosion -> thin
    #             img_af = cv2.erode(img_af, k, iterations=1)
    #         else:
    #             # dilation -> thicken
    #             img_af = cv2.dilate(img_af, k, iterations=1)

    #     # -------------------------
    #     # 4) small gaussian noise (fixed, correct float handling)
    #     p_noise = 0.25
    #     if random.random() < p_noise:
    #         # convert to float, add gaussian noise, clip back
    #         img_f = img_af.astype(np.float32)
    #         sigma = 2.0  # small sigma preserves stroke detail
    #         noise = np.random.normal(0.0, sigma, img_f.shape).astype(np.float32)
    #         img_f = img_f + noise
    #         img_f = np.clip(img_f, 0.0, 255.0)
    #         img_af = img_f.astype(np.uint8)

    #     # -------------------------
    #     # 5) optional small blur (very mild)
    #     if random.random() < 0.12:
    #         r = random.uniform(0.2, 0.6)  # radius
    #         # OpenCV GaussianBlur expects kernel size; convert radius -> ksize (odd)
    #         k = max(3, int(r * 3) | 1)
    #         img_af = cv2.GaussianBlur(img_af, (k, k), sigmaX=r, borderType=cv2.BORDER_DEFAULT)

    #     # -------------------------
    #     # final downsample to target size with INTER_AREA to reduce aliasing
    #     img_out = cv2.resize(img_af, (w, h), interpolation=cv2.INTER_AREA)
    #     h, w = img.shape[:2]
    #     return img_out
    
    # def _prepare_style_augment(self, img):
    #     """
    #     溫和版增強：依靠遮擋來破壞 Content，但保護幾何風格與粗細。
    #     """
    #     h, w = img.shape  # 獲取原圖尺寸
        
    #     # 1. 輕微縮放 (Scale) - 安全
    #     scale = random.uniform(0.9, 1.0) 
    #     crop_h, crop_w = int(h * scale), int(w * scale)
        
    #     # 避免裁切尺寸為 0
    #     crop_h = max(1, crop_h)
    #     crop_w = max(1, crop_w)

    #     start_y = random.randint(0, h - crop_h)
    #     start_x = random.randint(0, w - crop_w)
        
    #     img_crop = img[start_y:start_y+crop_h, start_x:start_x+crop_w]
        
    #     # Resize 回原本的大小 (w, h) -> OpenCV 格式為 (寬, 高)
    #     img_aug = cv2.resize(img_crop, (w, h))

    #     # 2. 非常輕微的幾何變換 - 保護風格
    #     # 角度控制在 +/- 5 度以內
    #     angle = random.uniform(-5, 5)  
    #     tx = random.uniform(-2, 2)
    #     ty = random.uniform(-2, 2)
        
    #     # 中心點改為 (w/2, h/2)
    #     M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    #     M[0, 2] += tx
    #     M[1, 2] += ty
        
    #     # 變換後維持原圖大小 (w, h)
    #     img_aug = cv2.warpAffine(img_aug, M, (w, h), 
    #                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    #     # 3. [主力] Random Erasing (Cutout)
    #     if random.random() < 0.8: # 80% 機率觸發
    #         # 改為相對比例：遮擋長寬約為原圖的 20% ~ 40%
    #         min_dim = min(h, w)
    #         min_erase = int(min_dim * 0.2)
    #         max_erase = int(min_dim * 0.4)
            
    #         # 確保有合理的遮擋範圍
    #         if max_erase > min_erase and max_erase > 0:
    #             erase_w = random.randint(min_erase, max_erase)
    #             erase_h = random.randint(min_erase, max_erase)
                
    #             erase_x = random.randint(0, w - erase_w)
    #             erase_y = random.randint(0, h - erase_h)
                
    #             img_aug[erase_y:erase_y+erase_h, erase_x:erase_x+erase_w] = 255 

    #     return img_aug

    def _prepare_style_augment(self, img):
        """
        激進版增強：Patch Shuffle (拼圖打亂) + 強雜訊 + 遮擋
        目標：徹底破壞文字結構，只保留局部筆觸紋理。
        """
        h, w = img.shape
        img_aug = img.copy()

        # =================================================================
        # 1. [新核心] Random Patch Shuffle (隨機拼圖)
        # 將圖片切成 grid_size x grid_size 的網格並打亂
        # 這能最有效防止模型偷看文字內容
        # =================================================================
        if random.random() < 0.8: # 80% 機率觸發打亂
            grid_size = 2 # 切成 2x2 = 4 塊 (每塊 32x32)
            # 如果圖片太小，就不切
            if h >= 16 and w >= 16:
                h_step = h // grid_size
                w_step = w // grid_size
                
                patches = []
                for i in range(grid_size):
                    for j in range(grid_size):
                        y1, y2 = i*h_step, (i+1)*h_step
                        x1, x2 = j*w_step, (j+1)*w_step
                        patches.append(img[y1:y2, x1:x2].copy())
                
                # 打亂拼圖
                random.shuffle(patches)
                
                # 拼回去
                idx = 0
                for i in range(grid_size):
                    for j in range(grid_size):
                        y1, y2 = i*h_step, (i+1)*h_step
                        x1, x2 = j*w_step, (j+1)*w_step
                        # Resize patch 回去填補 (防止維度微小誤差)
                        patch = cv2.resize(patches[idx], (x2-x1, y2-y1))
                        img_aug[y1:y2, x1:x2] = patch
                        idx += 1

        # =================================================================
        # 2. 隨機縮放與幾何 (輕微，保持筆觸特性)
        # =================================================================
        # 既然結構已經被打亂，幾何變換可以稍微輕一點，避免破壞筆觸方向性
        if random.random() < 0.5:
            angle = random.uniform(-5, 5)  
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img_aug = cv2.warpAffine(img_aug, M, (w, h), 
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        # =================================================================
        # 3. 強雜訊 (Gaussian Noise) - 解決 NCE Loss 過低
        # 增加雜訊讓 Style Encoder 更難比對像素，必須學特徵
        # =================================================================
        if random.random() < 0.6:
            # 雜訊強度 15~40 (視圖片灰階值而定，通常 0-255)
            sigma = random.randint(15, 40)
            noise = np.random.normal(0, sigma, img_aug.shape)
            img_aug = img_aug.astype(np.float32) + noise
            img_aug = np.clip(img_aug, 0, 255).astype(np.uint8)

        # =================================================================
        # 4. 隨機遮擋 (Random Erasing) - 輔助 Patch Shuffle
        # =================================================================
        if random.random() < 0.5:
            min_dim = min(h, w)
            erase_size = random.randint(int(min_dim*0.2), int(min_dim*0.4))
            ex = random.randint(0, w - erase_size)
            ey = random.randint(0, h - erase_size)
            img_aug[ey:ey+erase_size, ex:ex+erase_size] = 255 # 填白

        return img_aug
    # ================= [修改] get_style_ref =================
    def get_style_ref(self, image_path, laplace_path):
        """
        修改版：接收圖片路徑，進行單圖自監督增強
        """
        # 1. 讀取原始圖片
        style_img_raw = cv2.imread(image_path, flags=0)
        # 讀取或建立 Laplace 圖 (若無對應檔案則用全黑替代，避免報錯)
        laplace_img_raw = cv2.imread(laplace_path, flags=0)
        
        if style_img_raw is None:
             raise ValueError(f"Image not found: {image_path}")
        if laplace_img_raw is None:
            laplace_img_raw = np.zeros_like(style_img_raw)

        # 2. 產生兩個不同的視圖 (View 1 & View 2)
        # 透過兩次呼叫 _prepare_style_augment，得到兩個隨機變形版本
        style_1 = self._prepare_style_augment(style_img_raw)
        style_2 = self._prepare_style_augment(style_img_raw)
        
        # Laplace 圖通常不適合做複雜幾何變換，這裡簡單 Resize 配合
        laplace_1 = laplace_transform(style_1)
        laplace_2 = laplace_transform(style_2)

        # 3. 歸一化與堆疊
        # 依照原始邏輯，這裡只除以 255.0
        style_images = np.array([style_1, style_2]).astype(np.float32) / 255.0
        laplace_images = np.array([laplace_1, laplace_2]).astype(np.float32) / 255.0

        return style_images, laplace_images

    # def get_style_ref(self, wr_id):
    #     style_list = os.listdir(os.path.join(self.style_path, wr_id))
    #     style_index = random.sample(range(len(style_list)), 2) # anchor and positive
    #     style_images = [cv2.imread(os.path.join(self.style_path, wr_id, style_list[index]), flags=0)
    #                     for index in style_index]
    #     laplace_images = [cv2.imread(os.path.join(self.laplace_path, wr_id, style_list[index]), flags=0)
    #                       for index in style_index]
        
    #     height = style_images[0].shape[0]
    #     assert height == style_images[1].shape[0], 'the heights of style images are not consistent'
    #     max_w = max([style_image.shape[1] for style_image in style_images])
        
    #     '''style images'''
    #     style_images = [style_image/255.0 for style_image in style_images]
    #     new_style_images = np.ones([2, height, max_w], dtype=np.float32)
    #     new_style_images[0, :, :style_images[0].shape[1]] = style_images[0]
    #     new_style_images[1, :, :style_images[1].shape[1]] = style_images[1]

    #     '''laplace images'''
    #     laplace_images = [laplace_image/255.0 for laplace_image in laplace_images]
    #     new_laplace_images = np.zeros([2, height, max_w], dtype=np.float32)
    #     new_laplace_images[0, :, :laplace_images[0].shape[1]] = laplace_images[0]
    #     new_laplace_images[1, :, :laplace_images[1].shape[1]] = laplace_images[1]
    #     return new_style_images,new_laplace_images
    
    def get_symbols(self, input_type):
        pkl_path = os.path.join(self.root, f"{input_type}.pickle")
        with open(pkl_path, "rb") as f:
            symbols = pickle.load(f)

        symbols = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in symbols}
        contents = []
        for char in self.letters:
            symbol = torch.from_numpy(symbols[ord(char)]).float()
            contents.append(symbol)
        contents.append(torch.zeros_like(contents[0])) # blank image as PAD_TOKEN
        contents = torch.stack(contents)
        return contents
       
    def __len__(self):
        return len(self.indices)

    ### Borrowed from GANwriting ###
    def label_padding(self, labels, max_len):
        ll = [self.letter2index[i] for i in labels]
        num = max_len - len(ll)
        if not num == 0:
            ll.extend([self.tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
        return ll

# ================= [修改] __getitem__ =================
    def __getitem__(self, idx):
        image_name = self.data_dict[self.indices[idx]]['image']
        label = self.data_dict[self.indices[idx]]['label']
        wr_id = self.data_dict[self.indices[idx]]['s_id'] # 取得 ID (僅供路徑使用，不參與訓練邏輯)
        
        # 組合路徑 (請依據您的資料夾結構調整)
        # 假設結構為: root/image/split/wr_id/image.png
        img_path = os.path.join(self.image_path, wr_id, image_name)
        
        # 風格圖與 Laplace 圖的路徑通常與原圖相同
        style_path_full = os.path.join(self.style_path, wr_id, image_name)
        laplace_path_full = os.path.join(self.laplace_path, wr_id, image_name)

        # 讀取 Ground Truth (Content)
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)

        # [關鍵修改] 傳入路徑，進行單圖增強
        style_ref, laplace_ref = self.get_style_ref(style_path_full, laplace_path_full)
        
        # 轉 Tensor
        style_ref = torch.from_numpy(style_ref).to(torch.float32)
        laplace_ref = torch.from_numpy(laplace_ref).to(torch.float32)

        # 偽 ID (讓 Trainer 不報錯即可)
        try:
            wid = int(wr_id)
        except ValueError:
            wid = 0
        return {'img':image,
                'content':label, 
                'style':style_ref,
                "laplace":laplace_ref,
                'wid':wid,
                'transcr':label,
                'image_name':image_name}

    # def __getitem__(self, idx):
    #     image_name = self.data_dict[self.indices[idx]]['image']
    #     label = self.data_dict[self.indices[idx]]['label']
    #     wr_id = self.data_dict[self.indices[idx]]['s_id']
    #     transcr = label
    #     img_path = os.path.join(self.image_path, wr_id, image_name)
    #     image = Image.open(img_path).convert('RGB')
    #     image = self.transforms(image)

    #     style_ref, laplace_ref = self.get_style_ref(wr_id)
    #     style_ref = torch.from_numpy(style_ref).to(torch.float32) # [2, h , w] achor and positive
    #     laplace_ref = torch.from_numpy(laplace_ref).to(torch.float32) # [2, h , w] achor and positive

    #     try:
    #         wid = int(wr_id)
    #     except ValueError:
    #         wid = 0
    #     return {'img':image,
    #             'content':label, 
    #             'style':style_ref,
    #             "laplace":laplace_ref,
    #             'wid':wid,
    #             'transcr':label,
    #             'image_name':image_name}


    def collate_fn_(self, batch):
        width = [item['img'].shape[2] for item in batch]
        c_width = [len(item['content']) for item in batch]
        s_width = [item['style'].shape[2] for item in batch]

        transcr = [item['transcr'] for item in batch]
        target_lengths = torch.IntTensor([len(t) for t in transcr])
        image_name = [item['image_name'] for item in batch]

        if max(s_width) < self.style_len:
            max_s_width = max(s_width)
        else:
            max_s_width = self.style_len

        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], max(width)], dtype=torch.float32)
        content_ref = torch.zeros([len(batch), max(c_width), 16 , 16], dtype=torch.float32)
        
        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width], dtype=torch.float32)
        laplace_ref = torch.zeros([len(batch), batch[0]['laplace'].shape[0], batch[0]['laplace'].shape[1], max_s_width], dtype=torch.float32)
        target = torch.zeros([len(batch), max(target_lengths)], dtype=torch.int32)

        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print('img', item['img'].shape)
            try:
                content = [self.letter2index[i] for i in item['content']]
                content = self.con_symbols[content]
                content_ref[idx, :len(content)] = content
            except:
                print('content', item['content'])

            target[idx, :len(transcr[idx])] = torch.Tensor([self.letter2index[t] for t in transcr[idx]])
            
            try:
                if max_s_width < self.style_len:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace']
                else:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :self.style_len]
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace'][:, :, :self.style_len]
            except:
                print('style', item['style'].shape)

        wid = torch.tensor([item['wid'] for item in batch])
        content_ref = 1.0 - content_ref # invert the image
        return {'img':imgs, 'style':style_ref, 'content':content_ref, 'wid':wid, 'laplace':laplace_ref,
                'target':target, 'target_lengths':target_lengths, 'image_name':image_name}


class Random_StyleDataset(HandwritingDataset):
    def __init__(self, ref_num,split, cfg=None) -> None:
        super().__init__(cfg=cfg, split=split)
        
        # [新增] 簡單判斷：若 RandomStyle 找不到資料，試著去找 train 資料夾
        # 這是為了 test.py 中使用 'iv_s' 等情況，通常對應到 'train'
        if not os.path.exists(self.style_path) or not os.listdir(self.style_path):
             if os.path.isdir(os.path.join(self.style_path, 'train')):
                 self.style_path = os.path.join(self.style_path, 'train')
                 self.laplace_path = os.path.join(self.laplace_path, 'train')

        if os.path.exists(self.style_path):
            self.author_id = os.listdir(self.style_path)
        else:
            self.author_id = []
            
        self.ref_num = ref_num
    
    def __len__(self):
        return self.ref_num
    
    def get_style_ref(self, wr_id): # Choose the style image whose length exceeds 32 pixels
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        random.shuffle(style_list)
        for index in range(len(style_list)):
            style_ref = style_list[index]

            style_image = cv2.imread(os.path.join(self.style_path, wr_id, style_ref), flags=0)
            laplace_image = cv2.imread(os.path.join(self.laplace_path, wr_id, style_ref), flags=0)
            if style_image.shape[1] > 128:
                break
            else:
                continue
        style_image = style_image/255.0
        laplace_image = laplace_image/255.0
        return style_image, laplace_image

    def __getitem__(self, _):
        batch = []
        for idx in self.author_id:
            style_ref, laplace_ref = self.get_style_ref(idx)
            style_ref = torch.from_numpy(style_ref).unsqueeze(0)
            style_ref = style_ref.to(torch.float32)
            laplace_ref = torch.from_numpy(laplace_ref).unsqueeze(0)
            laplace_ref = laplace_ref.to(torch.float32)
            wid = idx
            batch.append({'style':style_ref, 'laplace':laplace_ref, 'wid':wid})
        
        s_width = [item['style'].shape[2] for item in batch]
        if max(s_width) < self.style_len:
            max_s_width = max(s_width)
        else:
            max_s_width = self.style_len
        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width], dtype=torch.float32)
        laplace_ref = torch.zeros([len(batch), batch[0]['laplace'].shape[0], batch[0]['laplace'].shape[1], max_s_width], dtype=torch.float32)
        wid_list = []
        for idx, item in enumerate(batch):
            try:
                if max_s_width < self.style_len:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace']
                else:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :self.style_len]
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace'][:, :, :self.style_len]
                wid_list.append(item['wid'])
            except:
                print('style', item['style'].shape)
        
        return {'style':style_ref, 'laplace':laplace_ref,'wid':wid_list}

class ContentData(HandwritingDataset):
    def __init__(self,split='train', content_type='unifont', cfg=None) -> None:
        super().__init__(cfg=cfg, split=split, content_type=content_type)
       
    def get_content(self, label):
        word_arch = [self.letter2index[i] for i in label]
        content_ref = self.con_symbols[word_arch]
        content_ref = 1.0 - content_ref
        return content_ref.unsqueeze(0)