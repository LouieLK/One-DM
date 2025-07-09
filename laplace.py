import torch, cv2
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

laplace = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

def laplace_ostu(file):
    image = cv2.imread(file, 1)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img.transpose([2, 0, 1])).unsqueeze(0).float()
    y = F.conv2d(x, laplace.repeat(1, 3, 1, 1), stride=1, padding=1, )
    y = y.squeeze().numpy()
    y = np.clip(y, 0, 255)
    y = y.astype(np.uint8)
    ret, threshold = cv2.threshold(y, 0, 255, cv2.THRESH_OTSU)
    return threshold

def batch_laplace(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                input_path = os.path.join(root, file)
                output_root = root.replace(input_dir, output_dir)
                if not os.path.exists(output_root):
                    os.makedirs(output_root)
                output_path = os.path.join(output_root, file)
                threshold = laplace_ostu(input_path)
                cv2.imwrite(output_path, threshold)

if __name__ == '__main__':
    input_dir = '/One-DM/data/Traditional-Chinese-Handwriting-Dataset/data'
    output_dir = '/One-DM/data/Traditional-Chinese-Handwriting-Dataset/data_laplace'
    batch_laplace(input_dir, output_dir)