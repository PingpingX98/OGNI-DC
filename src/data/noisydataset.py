import torch
import numpy as np
import random
from torch.utils.data import Dataset
import cv2


class NoisyDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        print(f"base dataset is {self.base_dataset}")
        self.noise_types = ['gaussian', 'impulse', 'rayleigh', 'gamma', 'exponential', 'uniform']

    def __len__(self):
        return len(self.base_dataset)

    def add_noise(self, x, noise_type = 'gaussian'):

        if noise_type == 'gaussian':
            std = random.uniform(0.01, 0.2)
            noise = torch.randn_like(x) * std
            x_noisy = x + noise
            return x_noisy.clamp(0, 1), f'gaussian(std={std:.3f})'

        elif noise_type == 'impulse': 
            prob = random.uniform(0.01, 0.05)
            x_noisy = x.clone()
            rand = torch.rand_like(x_noisy)
            x_noisy[rand < (prob / 2)] = 0.0
            x_noisy[rand > 1 - (prob / 2)] = 1.0
            return x_noisy, f'impulse(prob={prob:.3f})'

        elif noise_type == 'rayleigh':
            scale = random.uniform(0.1, 0.5)
            rayleigh_noise = torch.from_numpy(np.random.rayleigh(scale=scale, size=x.shape)).float()
            x_noisy = x + rayleigh_noise
            return x_noisy.clamp(0, 1), f'rayleigh(scale={scale:.3f})'

        elif noise_type == 'gamma':
            shape = random.uniform(1.0, 5.0)
            gamma_noise = torch.from_numpy(np.random.gamma(shape, scale=1.0, size=x.shape)).float()
            x_noisy = x + gamma_noise / 10.0  
            return x_noisy.clamp(0, 1), f'gamma(shape={shape:.2f})'

        elif noise_type == 'exponential':
            lam = random.uniform(1.0, 10.0)
            exp_noise = torch.from_numpy(np.random.exponential(scale=1.0 / lam, size=x.shape)).float()
            x_noisy = x + exp_noise
            return x_noisy.clamp(0, 1), f'exponential(lam={lam:.2f})'

        elif noise_type == 'uniform':
            low = random.uniform(-0.2, 0.0)
            high = random.uniform(0.0, 0.2)
            uniform_noise = torch.empty_like(x).uniform_(low, high)
            x_noisy = x + uniform_noise
            return x_noisy.clamp(0, 1), f'uniform({low:.2f}, {high:.2f})'
        
    def load_image_cv2(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR) 
        
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        img = img.astype(np.float32) / 255.0  
        img = np.transpose(img, (2, 0, 1))   
        tensor = torch.from_numpy(img)
        return tensor

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

 
        if isinstance(item, dict):
            x = item['rgb']
            y = item['dep']

        elif isinstance(item, (tuple, list)):
            x, y = item[0], item[1]

        else:
            raise TypeError(f"Unsupported item type: {type(item)}")

        if isinstance(x, torch.ByteTensor) or (isinstance(x, torch.Tensor) and x.max() > 1.0):
            x = x.float() / 255.0

        if x.ndim == 2:
            x = x.unsqueeze(0)

        x_noisy, noise_info = self.add_noise(x)
        return {
            "rgb": x_noisy,
            "depth": y,
        }

    
    """ def __getitem__(self, idx):
        x, y, *_= self.base_dataset[idx]

        if isinstance(x, np.ndarray):
            x = torch.tensor(x / 255.0, dtype=torch.float32)
        if isinstance(x, torch.ByteTensor):
            x = x.float() / 255.0
        if x.ndim == 2:
            x = x.unsqueeze(0)

        x_noisy, noise_info = self.add_noise(x)
        return x_noisy, y, noise_info """
