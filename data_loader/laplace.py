import torch
import torch.nn.functional as F
import numpy as np
import cv2

# 定義 Laplace Kernel (保持與原檔案一致)
# Shape: [1, 1, 3, 3] -> 1個輸出通道, 1個輸入通道, 3x3 kernel
laplace_kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

def laplace_transform(img_np):
    """
    針對單張 numpy 圖片進行 Laplace 邊緣提取與 Otsu 二值化。
    
    Args:
        img_np (numpy.ndarray): 輸入圖片，支援灰階 (H, W) 或 (H, W, 1)。
                                建議使用 cv2.imread(path, flags=0) 讀取。
    Returns:
        threshold (numpy.ndarray): 處理後的邊緣二值化圖片 (H, W)。
    """
    # 1. 轉換為 Tensor 並調整維度
    # 輸入如果是灰階 (H, W)，需要擴充為 (1, 1, H, W) 以符合 F.conv2d 的輸入要求 [Batch, Channel, Height, Width]
    if img_np.ndim == 2:
        x = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float()
    elif img_np.ndim == 3:
        # 如果輸入是 (H, W, 1)，轉為 (1, 1, H, W)
        x = torch.from_numpy(img_np.transpose((2, 0, 1))).unsqueeze(0).float()
    else:
        raise ValueError(f"不支援的圖片維度: {img_np.shape}")

    # 2. 執行卷積 (Convolution)
    # 因為輸入是單通道灰階，直接使用 [1, 1, 3, 3] 的 kernel 即可
    # padding=1 確保輸出圖片大小不變
    y = F.conv2d(x, laplace_kernel, stride=1, padding=1)

    # 3. 後處理 (轉回 Numpy -> Clip -> uint8)
    y = y.squeeze().numpy() # 去除 Batch 和 Channel 維度 -> (H, W)
    y = np.clip(y, 0, 255)  # 確保數值在 0-255 之間
    y = y.astype(np.uint8)

    # 4. Otsu 二值化 (這是原始程式碼的關鍵步驟)
    # 將灰階邊緣圖轉為黑白二值圖，強化特徵
    ret, threshold = cv2.threshold(y, 0, 255, cv2.THRESH_OTSU)
    
    return threshold