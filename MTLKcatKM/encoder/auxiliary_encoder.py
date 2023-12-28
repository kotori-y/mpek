import numpy as np

from MTLKcatKM.utils import OrganismTokenizer

import torch
from torch import nn


class RBF(nn.Module):
    """
    Radial Basis Function
    """

    def __init__(self, centers, gamma, device=None):
        super(RBF, self).__init__()
        self.device = device
        if device is None:
            self.device = torch.device('cpu')

        self.centers = torch.reshape(torch.tensor(centers, dtype=torch.float32), [1, -1]).to(self.device)
        self.gamma = gamma

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = torch.reshape(x, [-1, 1])
        return torch.exp(-self.gamma * torch.square(x - self.centers)).to(self.device)


class ConditionFloatRBF(nn.Module):
    """
    Bond Float Encoder using Radial Basis Functions
    """
    def __init__(self, embed_dim, device=None):
        super(ConditionFloatRBF, self).__init__()
        self.device = device
        if device is None:
            self.device = torch.device('cpu')

        self.condition_names = ["pH", "temperature"]
        self.rbf_params = {
            'pH': (np.arange(0, 14, 0.1), 10.0),
            'temperature': (np.arange(0, 100, 1), 10.0)
        }

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()

        for name in self.condition_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma, device=self.device)

            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

    def forward(self, condition):
        """
        Args:
            bond_float_features(dict of tensor): bond float features.
        """
        out_embed = 0
        for i, name in enumerate(self.condition_names):
            if name in condition:
                x = condition[name]
                if x is not None:
                    rbf_x = self.rbf_list[i](x)
                    out_embed += self.linear_list[i](rbf_x)
        return out_embed


class AuxiliaryEncoder(nn.Module):
    def __init__(self, organism_dict_size: int, embed_size: int, device,
                 use_ph=True, use_temperature=True, use_organism=True):
        super().__init__()
        self.embed_dim = embed_size

        self.use_ph = use_ph
        self.use_temperature = use_temperature
        self.use_organism = use_organism

        # self.dropout = nn.Dropout(0.2)
        # self.scale = torch.sqrt(torch.FloatTensor([embed_size])).to(device)
        if use_organism:
            self.organism_embedding = nn.Embedding(organism_dict_size + 1, embed_size)

        if use_ph or use_temperature:
            self.cond_embedding = ConditionFloatRBF(embed_size, device=device)

    def forward(self, organism, condition=None):
        out = 0

        if self.use_organism:
            out += self.organism_embedding(organism)

        if self.use_ph or self.use_temperature:
            out += self.cond_embedding(condition)

        return out
