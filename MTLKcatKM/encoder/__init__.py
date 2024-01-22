import torch
from torch import nn
from torch.utils.data import RandomSampler
from torch_geometric.loader import DataLoader

from MTLKcatKM.encoder.ligand_encoder import LigandEncoderMoleBert
from MTLKcatKM.encoder.protein_encoder import ProteinEncoder
from MTLKcatKM.encoder.auxiliary_encoder import AuxiliaryEncoder
from MTLKcatKM.layers.attention import MultiHeadAttentionLayer
from MTLKcatKM.layers.mlp import MLP


class ComplexEncoder(nn.Module):
    def __init__(
            self, ligand_enc: LigandEncoderMoleBert, protein_enc: ProteinEncoder,
            auxiliary_enc: AuxiliaryEncoder, use_esm2=False,
            dropout=0.2, use_attention=False, use_ph=True, use_temperature=True, use_organism=True,
            atten_heads=16, device=None
    ):
        super().__init__()
        self.device = device
        self.use_esm2 = use_esm2

        if device is None:
            self.device = torch.device('cpu')

        self.ligand_enc = ligand_enc
        self.protein_enc = protein_enc

        self.use_ph = use_ph
        self.use_temperature = use_temperature
        self.use_organism = use_organism

        if self.use_ph or self.use_temperature or self.use_organism:
            self.auxiliary_enc = auxiliary_enc
        else:
            self.auxiliary_enc = None

        self.ligand_hidden = self.ligand_enc.embed_dim
        self.protein_hidden = self.protein_enc.embed_dim

        self.use_attention = use_attention
        if self.use_attention:
            self.attn_layer = MultiHeadAttentionLayer(
                hid_dim=self.protein_hidden,
                n_heads=atten_heads,
                dropout=dropout,
                device=self.device
            )
            self.attn_norm_layer = nn.LayerNorm(self.protein_hidden)
            self.ff_norm_layer = nn.LayerNorm(self.protein_hidden)

            self.ff_layer = MLP(self.protein_hidden, [self.protein_hidden * 2, self.protein_hidden], dropout=dropout)
            self.dropout_layer = nn.Dropout(dropout)
        
    def make_src_mask(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def forward(self, input_ids, attention_mask, mol_graph, organ_ids, condition):
        h_lig = self.ligand_enc(mol_graph)
        # print(h_lig, h_lig.shape)

        if not self.use_esm2:
            h_prot = self.protein_enc(input_ids, attention_mask)
        else:
            h_prot = self.protein_enc(input_ids)
        # h_prot = self.pro2lig(self.protein_enc(sequence))
        if self.auxiliary_enc:
            h_aux = self.auxiliary_enc(organ_ids, condition)
        else:
            h_aux = None

        return h_lig, h_prot, h_aux


if __name__ == "__main__":
    from MTLKcatKM.datasets import MTLKDataset

    ligand_model_path = "../pretrained/checkpoints/model.pth"
    protein_model_path = "../../ProtTrans/models/prot_t5_xl_uniref50"

    dataset_params = {
        "sequence_idx": 0,
        "smiles_idx": 1,
        "label_idx": [7, 8],
        "max_length": 512
    }

    train_dataset = MTLKDataset(data_path='../data/modeling/train_dataset.csv', model_name=protein_model_path, **dataset_params)
    sampler_train = RandomSampler(train_dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, 4, drop_last=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler_train,
        num_workers=10,
    )

    enc1 = LigandEncoder(ligand_model_path)
    enc2 = ProteinEncoder(protein_model_path)
    encoder = ComplexEncoder(ligand_enc=enc1, protein_enc=enc2)

    for step, batch in enumerate(train_loader):
        net_input = {
            "mol_graph": batch["mol_graph"],
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"]
        }

        net_outputs = encoder(**net_input)

        break
