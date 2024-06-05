from typing import List
from functools import reduce

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import RandomSampler
from torch.nn import BatchNorm1d

from MTLKcatKM.encoder import ComplexEncoder, LigandEncoderMoleBert, ProteinEncoder, AuxiliaryEncoder
from MTLKcatKM.layers import MMoE, MLP, PLE
from MTLKcatKM.layers.mlp import MLPwoLastAct
# from MTLKcatKM.layers.mmoe import MMoEModule

# from torcheval.metrics.functional import r2_score


class MTLModel(nn.Module):
    def __init__(
            self, ligand_enc: LigandEncoderMoleBert, protein_enc: ProteinEncoder, auxiliary_enc: AuxiliaryEncoder,
            expert_out: int, expert_hidden: int, expert_layers: int, num_experts: int,
            num_tasks: int, tower_hid_layer=5, tower_hid_unit=128, weights=[1, 1], tower_dropout=[0.2, 0.2],
            dropout=0.2, use_ph=True, use_temperature=True, use_organism=True,
            num_ple_layers=1, gen_vec=False, device=None,
    ):
        super().__init__()
        self.device = device
        if device is None:
            self.device = torch.device('cpu')

        self.num_tasks = num_tasks
        self.use_ph = use_ph
        self.use_temperature = use_temperature
        self.use_organism = use_organism
        self.gen_vec = gen_vec

        self.encoder = ComplexEncoder(
            ligand_enc=ligand_enc, protein_enc=protein_enc, auxiliary_enc=auxiliary_enc,
            use_ph=use_ph, use_temperature=use_temperature, use_organism=use_organism, device=self.device,
        )
        self.pro2lig = nn.Linear(protein_enc.embed_dim, ligand_enc.embed_dim)

        if self.use_ph or self.use_temperature or self.use_organism:
            input_size = ligand_enc.embed_dim * 3
        else:
            input_size = ligand_enc.embed_dim * 2

        self.weights = weights

        self.multi_block = PLE(
            experts_in=input_size,
            experts_out=expert_out,
            experts_hidden=expert_hidden,
            expert_hid_layer=expert_layers,
            dropout_rate=dropout,
            num_experts=num_experts,
            num_tasks=num_tasks,
            num_ple_layers=num_ple_layers
        )

        if not self.gen_vec:
            self.towers = nn.ModuleList(
                [
                    MLP(
                        in_size=expert_out,
                        hidden_size=tower_hid_unit,
                        out_size=1,
                        layer_num=tower_hid_layer,
                        dropout_rate=tower_dropout[i]
                    ) for i in range(num_tasks)
                ]
            )

        self.criterion = nn.MSELoss()

    def forward(self, input_ids, attention_mask, mol_graph, organ_ids, condition, task_names):
        # net_outputs = []

        h_lig, h_prot, h_aux = self.encoder(input_ids, attention_mask, mol_graph, organ_ids, condition)
        h_prot = self.pro2lig(h_prot)

        if self.use_ph or self.use_temperature or self.use_organism:
            h = torch.cat([h_lig, h_prot, h_aux], 1)  # [bs, 512 * 3]
        else:
            h = torch.cat([h_lig, h_prot], 1)  # [bs, 512 * 3]

        tower_input = self.multi_block(h)

        if self.gen_vec:
            return np.hstack([x.cpu().numpy() for x in tower_input])

        pred = [t(ti) for t, ti in zip(self.towers, tower_input)]
        return dict(zip(task_names, pred))

    def compute_loss(self, net_outputs, labels):
        weights_mapping = dict(zip(labels, self.weights))
        loss_dict = {}

        loss = 0

        # n = 0
        for task_name, target in labels.items():
            predict = net_outputs[task_name]

            mask = torch.isnan(target)
            # out = (predict[~mask] - target[~mask]) ** 2
            # out = torch.nan_to_num(out)
            # mse = out.mean()
            mask = ~mask
            target = torch.nan_to_num(target)
            # 
            # # if mask.all().detach().item():
            # #     tmp_loss = torch.sum(((predict - target) * mask) ** 2.0)
            # # else:
            mse = torch.sum(((predict - target) * mask) ** 2.0) / (torch.sum(mask) + 1e-5)
            # tmp_loss = self.criterion(predict * mask, target * mask)
            # tmp_loss = self.criterion(target, predict)
            # tmp_loss = self.criterion(predict, target)
            # print(task_name, weights_mapping)
            loss += mse * weights_mapping[task_name]
            loss_dict[f"{task_name}_loss"] = loss_dict.get(f"{task_name}_loss", 0) + mse.detach().item()

            # n += 1

        loss_dict["loss"] = loss.detach().item()

        return loss, loss_dict

    def compute_r2(self, net_outputs, labels):
        r2_dict = {}

        # n = 0
        for task_name, target in labels.items():
            predict = net_outputs[task_name].flatten()

            mask = torch.isnan(target)
            mask = ~mask
            target = torch.nan_to_num(target)

            # if mask.all().detach().item():
            #     mask = ~mask
            #     tmp_loss = torch.sum(((predict - target) * mask) ** 2.0)
            # else:
            #     mask = ~mask
            #     tmp_loss = torch.sum(((predict - target) * mask) ** 2.0) / torch.sum(mask)
            tmp_r2 = r2_score(predict * mask, target * mask)
            r2_dict[f"{task_name}_r2"] = tmp_r2.detach().item()

            # n += 1
        return r2_dict


if __name__ == "__main__":
    from MTLKcatKM.datasets import MTLKDataset
    from torch_geometric.loader import DataLoader

    ligand_model_path = "./pretrained/checkpoints/model.pth"
    protein_model_path = "./pretrained/prot_t5_xl_uniref50"

    enc1 = LigandEncoder(ligand_model_path)
    enc2 = ProteinEncoder(protein_model_path)

    _mmoe_hid_dim = 512

    model = MTLModel(enc1, enc2, _mmoe_hid_dim, num_experts=3, num_tasks=2)

    train_dataset = MTLKDataset(
        data_path="./data/modeling/train_dataset.csv", sequence_idx=0, smiles_idx=1, label_idx=[-5, -6],
        model_name=protein_model_path, max_length=256
    )
    sampler_train = RandomSampler(train_dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, 10, drop_last=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler_train,
        num_workers=10,
    )

    for _, batch in enumerate(train_loader):
        net_input = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "mol_graph": batch["mol_graph"]
        }

        o = model(**net_input)
        print(model.compute_loss(o, batch["labels"]))
        break
