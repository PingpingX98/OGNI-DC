import torch
import random
import numpy as np

def add_noise(x, noise_type='gaussian', mask=None):
    if mask is None:
        mask = x > 0

    if noise_type == 'gaussian':
        std = random.uniform(0.01, 0.2)
        noise = torch.randn_like(x) * std
        x_noisy = x.clone()
        x_noisy[mask] = x[mask] + noise[mask]
        # x_noisy = x + noise
        return x_noisy, f'gaussian(std={std:.3f})'

    elif noise_type == 'impulse':
        prob = random.uniform(0.01, 0.05)
        x_noisy = x.clone()
        max_dep = 10.0
        rand = torch.rand_like(x_noisy)
        max_mask = (rand > 1 - (prob / 2)) & mask
        zero_mask = (rand < (prob / 2)) & mask
        x_noisy[zero_mask] = 0.0
        x_noisy[max_mask] = max_dep
        return x_noisy, f'impulse(prob={prob:.3f})'

    elif noise_type == 'rayleigh':
        scale = random.uniform(0.1, 0.5)
        rayleigh_noise = torch.from_numpy(np.random.rayleigh(scale=scale, size=x.shape)).float()
        x_noisy = x.clone()
        x_noisy[mask] = x[mask] + rayleigh_noise[mask]
        return x_noisy, f'rayleigh(scale={scale:.3f})'

    elif noise_type == 'gamma':
        shape = random.uniform(1.0, 5.0)
        gamma_noise = torch.from_numpy(np.random.gamma(shape, scale=1.0, size=x.shape)).float()
        x_noisy = x.clone()
        x_noisy[mask] = x[mask] + gamma_noise [mask] / 10.0
        return x_noisy, f'gamma(shape={shape:.2f})'

    elif noise_type == 'exponential':
        lam = random.uniform(1.0, 10.0)
        exp_noise = torch.from_numpy(np.random.exponential(scale=1.0 / lam, size=x.shape)).float()
        x_noisy = x.clone()
        x_noisy[mask] = x[mask] + exp_noise[mask]
        return x_noisy, f'exponential(lam={lam:.2f})'

    elif noise_type == 'uniform':
        low = random.uniform(-0.2, 0.0)
        high = random.uniform(0.0, 0.2)
        uniform_noise = torch.empty_like(x).uniform_(low, high)
        x_noisy = x.clone()
        x_noisy[mask] = x[mask] + uniform_noise[mask]
        return x_noisy, f'uniform({low:.2f}, {high:.2f})'

    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
