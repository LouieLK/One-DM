import torch
import torch.nn as nn

class CouplingLayer(nn.Module):
    """
    Affine Coupling Layer for RealNVP.
    """
    def __init__(self, num_inputs, num_hidden, mask):
        super(CouplingLayer, self).__init__()
        self.num_inputs = num_inputs
        self.mask = mask

        self.scale_net = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_inputs),
            nn.Tanh() # Tanh for stability
        )
        self.translate_net = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_inputs)
        )

        # scale_net 的最後一層是 Tanh() 之前的 Linear (index 為 4)
        nn.init.zeros_(self.scale_net[4].weight)
        nn.init.zeros_(self.scale_net[4].bias)
        
        # translate_net 的最後一層是 Linear (index 為 4)
        nn.init.zeros_(self.translate_net[4].weight)
        nn.init.zeros_(self.translate_net[4].bias)
        
    def forward(self, x, mode='forward'):
        mask = self.mask.to(x.device)
        x1 = x * mask
        
        s = self.scale_net(x1) * (1 - mask)
        t = self.translate_net(x1) * (1 - mask)

        if mode == 'forward':
            # Style (s) -> Latent (z)
            y = x1 + (1 - mask) * (x * torch.exp(s) + t)
            log_det_jacobian = torch.sum(s, dim=1)
            return y, log_det_jacobian
        else:
            # Latent (z) -> Style (s)
            y = x1 + (1 - mask) * ((x - t) * torch.exp(-s))
            return y

class NormalizingFlow(nn.Module):
    def __init__(self, num_inputs=512, num_hidden=512, num_layers=16):
        super(NormalizingFlow, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            mask = torch.zeros(num_inputs)
            if i % 2 == 0: mask[::2] = 1
            else: mask[1::2] = 1
            self.layers.append(CouplingLayer(num_inputs, num_hidden, mask))

    def forward(self, x):
        """ Training: s -> z """
        log_det_sum = 0
        for layer in self.layers:
            x, log_det = layer(x, mode='forward')
            log_det_sum += log_det
        return x, log_det_sum

    def reverse(self, z):
        """ Generation: z -> s """
        for layer in reversed(self.layers):
            z = layer(z, mode='inverse')
        return z