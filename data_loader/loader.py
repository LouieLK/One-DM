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

    def __init__(self, cfg=None, split='train', content_type='kaifont'):
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
    
    def get_style_ref(self, wr_id):
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        style_index = random.sample(range(len(style_list)), 2) # anchor and positive
        style_images = [cv2.imread(os.path.join(self.style_path, wr_id, style_list[index]), flags=0)
                        for index in style_index]
        laplace_images = [cv2.imread(os.path.join(self.laplace_path, wr_id, style_list[index]), flags=0)
                          for index in style_index]
        
        height = style_images[0].shape[0]
        assert height == style_images[1].shape[0], 'the heights of style images are not consistent'
        max_w = max([style_image.shape[1] for style_image in style_images])
        
        '''style images'''
        style_images = [(style_image / 255.0 - 0.5) / 0.5 for style_image in style_images]
        new_style_images = np.ones([2, height, max_w], dtype=np.float32)
        new_style_images[0, :, :style_images[0].shape[1]] = style_images[0]
        new_style_images[1, :, :style_images[1].shape[1]] = style_images[1]

        '''laplace images'''
        laplace_images = [(laplace_image / 255.0 - 0.5) / 0.5 for laplace_image in laplace_images]
        new_laplace_images = np.zeros([2, height, max_w], dtype=np.float32)
        new_laplace_images[0, :, :laplace_images[0].shape[1]] = laplace_images[0]
        new_laplace_images[1, :, :laplace_images[1].shape[1]] = laplace_images[1]
        return new_style_images, new_laplace_images
    
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

    def __getitem__(self, idx):
        image_name = self.data_dict[self.indices[idx]]['image']
        label = self.data_dict[self.indices[idx]]['label']
        wr_id = self.data_dict[self.indices[idx]]['s_id']
        transcr = label
        img_path = os.path.join(self.image_path, wr_id, image_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)

        style_ref, laplace_ref = self.get_style_ref(wr_id)
        style_ref = torch.from_numpy(style_ref).to(torch.float32) # [2, h , w] achor and positive
        laplace_ref = torch.from_numpy(laplace_ref).to(torch.float32) # [2, h , w] achor and positive

        return {'img':image,
                'content':label, 
                'style':style_ref,
                "laplace":laplace_ref,
                'wid':int(wr_id),
                'transcr':transcr,
                'image_name':image_name}


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
        c_h, c_w = self.con_symbols.shape[-2], self.con_symbols.shape[-1]
        content_ref = torch.zeros([len(batch), max(c_width), c_h , c_w], dtype=torch.float32)
        
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
        style_image = (style_image / 255.0 - 0.5) / 0.5
        laplace_image = (laplace_image / 255.0 - 0.5) / 0.5
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
    def __init__(self,split='train', content_type='kaifont', cfg=None) -> None:
        super().__init__(cfg=cfg, split=split, content_type=content_type)
       
    def get_content(self, label):
        word_arch = [self.letter2index[i] for i in label]
        content_ref = self.con_symbols[word_arch]
        content_ref = 1.0 - content_ref
        return content_ref.unsqueeze(0)