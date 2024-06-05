import torch
from torch import nn

import sys
sys.path.append("../..")
from Mole_BERT.model import GNN_graphpred

MODEL_CONFIG = {
    "num_layer": 5,  # number of graph conv layers
    "emb_dim": 300,  # embedding dimension in graph conv layers
    "feat_dim": 512,  # output feature dimention
    "drop_ratio": 0.0,  # dropout ratio
    "pool": "mean"
}


class LigandEncoderMoleBert(nn.Module):
    def __init__(self, init_model=None, device=None, frozen_params=False):
        super().__init__()
        self.embed_dim = 300
        self.frozen_params = frozen_params

        self.device = device
        if device is None:
            self.device = torch.device('cpu')

        self.encoder = GNN_graphpred(5, 300, drop_ratio=0.5)
        if init_model is not None and init_model != "":
            self.encoder.from_pretrained(init_model, device=self.device)
            # print("Loaded pre-trained model with success.")

        if frozen_params:
            # print(f"frozen {self}")
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            for name, p in self.encoder.named_parameters():
                if "encoder.gnn.batch_norms.4" not in name:
                    p.requires_grad = False
            # print(f"finetune {self}")

    def forward(self, mol_graph):
        if self.frozen_params:
            self.encoder.eval()
            with torch.no_grad():
                return self.encoder(mol_graph)
        return self.encoder(mol_graph)

    def __str__(self):
        return "MoleBert Ligand Encoder"


if __name__ == "__main__":

    # enc = LigandEncoder(init_model="../pretrained/checkpoints/model.pth", frozen_params=True)
    enc = LigandEncoderMoleBert(init_model="/fs1/home/wangll/software/Mole_BERT/model_gin/Mole-BERT.pth", frozen_params=True)
    n = 0
    for name, _ in enc.named_parameters():
        print(name)
        n += 1
    print(n)
