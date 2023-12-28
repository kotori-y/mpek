import argparse
import gc
import json
import os
import pickle
from collections import defaultdict
import torch.distributed as dist

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DistributedSampler, RandomSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import sys

sys.path.append("../")
from MTLKcatKM.datasets import MTLKDataset
from MTLKcatKM.encoder import LigandEncoder, ProteinEncoder, AuxiliaryEncoder, LigandEncoderMoleBert, ProteinEncoderESM2
from MTLKcatKM.model import MTLModel
from MTLKcatKM.utils import exempt_parameters, get_device, init_distributed_mode, TrainNormalizer, show_me_grad, \
    WarmCosine, ChildTuningAdamW

import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score, mean_squared_error


def gen_feat(model: MTLModel, loader, device, args):
    model.eval()
    pbar = tqdm(loader, desc="Iteration", disable=args.disable_tqdm)

    feat = []

    for step, batch in enumerate(pbar):
        net_input = {
            "mol_graph": batch["mol_graph"].to(device),
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device) if not args.use_esm2 else None,
            "organ_ids": batch["organ_ids"].to(device),
            "condition": {k: v.to(device) for k, v in batch["condition"].items()}
            # "condition": None
        }
        with torch.no_grad():
            labels = list(batch["labels"].keys())
            net_outputs = model(**net_input, task_names=labels)
            feat.append(net_outputs)

    return np.vstack(feat)


def main(args):
    device = get_device(args.device, args.local_rank)
    # device = get_device(args.device)

    if not args.use_esm2:
        prot_enc = ProteinEncoder(model_dir=args.prottrans_path, device=device, frozen_params=args.frozen_prot_enc)
    else:
        prot_enc = ProteinEncoderESM2(esm_dir=args.esm_dir, device=device, frozen_params=args.frozen_prot_enc)

    if args.use_molebert:
        lig_enc = LigandEncoderMoleBert(init_model=args.molebert_dir, device=device, frozen_params=args.frozen_ligand_enc)
    else:
        lig_enc = LigandEncoder(init_model=args.molclr_path, device=device, frozen_params=args.frozen_ligand_enc)

    with open(args.organism_dictionary_path) as f:
        _dict = json.load(f)

    aux_enc = AuxiliaryEncoder(
        organism_dict_size=len(_dict),
        embed_size=lig_enc.embed_dim,
        # embed_size=lig_enc.embed_dim + prot_enc.embed_dim,
        device=device
    )

    model = MTLModel(
        ligand_enc=lig_enc, protein_enc=prot_enc, auxiliary_enc=aux_enc,
        expert_out=args.expert_out, expert_hidden=args.expert_hidden,
        expert_layers=args.expert_layers, num_experts=args.num_experts,
        tower_hid_layer=args.tower_hid_layer, tower_hid_unit=args.tower_hid_unit,
        num_tasks=args.num_tasks, dropout=args.dropout, use_attention=args.use_attention,
        atten_heads=args.atten_heads, device=device, use_esm2=args.use_esm2, use_add=args.use_add, use_aux=args.use_aux,
        weights=args.weights, tower_dropout=args.tower_dropout, use_ple=args.use_ple, num_ple_layers=args.ple_layers, gen_vec=True
    ).to(device)
    gc.collect()

    model_without_ddp = model
    args.disable_tqdm = False

    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"#Params: {num_params}")

    dataset_params = {
        "organism_dictionary_path": args.organism_dictionary_path,
        # "sequence_index": args.sequence_index,
        "sequence_index": args.sequence_index,
        # "sequence_index_end": args.sequence_index_end,
        "smiles_index": args.smiles_index,
        "label_index": args.label_index,
        "max_length": args.max_length,
        "organism_index": args.organism_index,
        "ph_index": args.ph_index,
        "temperature_index": args.temperature_index,
        "use_esm2": args.use_esm2,
        "alphabet": prot_enc.alphabet if args.use_esm2 and args.esm_dir else None,
    }

    train_dataset = MTLKDataset(data_path=args.train_path, model_name=args.prottrans_path, **dataset_params)
    valid_dataset = MTLKDataset(data_path=args.valid_path, model_name=args.prottrans_path,
                                normalizer=train_dataset.normalizer, **dataset_params)
    test_dataset = MTLKDataset(data_path=args.test_path, model_name=args.prottrans_path,
                               normalizer=train_dataset.normalizer, **dataset_params)


    sampler_train = RandomSampler(train_dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler_train,
        num_workers=args.num_workers,
        shuffle=False
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("Generating...")
    train_feat = gen_feat(model, train_loader, device=device, args=args)
    valid_feat = gen_feat(model, valid_loader, device=device, args=args)
    test_feat = gen_feat(model, test_loader, device=device, args=args)
    print(valid_feat)

    with open(f'{args.out_dir}/train.npy', 'wb') as f:
        np.save(f, train_feat)

    with open(f'{args.out_dir}/valid.npy', 'wb') as f:
        np.save(f, valid_feat)

    with open(f'{args.out_dir}/test.npy', 'wb') as f:
        np.save(f, test_feat)


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--lr_warmup", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--checkpoint_dir", type=str, default="./model_params")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--train_path", type=str)
    parser.add_argument("--valid_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--out_dir", type=str, default='curves')

    parser.add_argument("--organism_dictionary_path", type=str)
    # parser.add_argument("--sequence_index", type=int, default=0)
    parser.add_argument("--sequence_index", type=int, default=0)
    # parser.add_argument("--sequence_index_end", type=int, default=0)
    parser.add_argument("--smiles_index", type=int, default=1)
    parser.add_argument("--organism_index", type=int)
    parser.add_argument("--ph_index", type=int)
    parser.add_argument("--temperature_index", type=int)
    parser.add_argument("--label_index", type=int, nargs="+")
    parser.add_argument("--weights", type=float, nargs="+")

    parser.add_argument("--max_length", type=int, default=4)

    parser.add_argument("--molclr_path", type=str)
    parser.add_argument("--prottrans_path", type=str)
    parser.add_argument("--esm_dir", type=str, default=None)
    parser.add_argument("--molebert_dir", type=str, default=None)
    parser.add_argument("--atten_heads", type=int, default=16)
    parser.add_argument("--use_attention", action="store_true", default=False)

    parser.add_argument("--tower_hid_layer", type=int, default=1)
    parser.add_argument("--tower_hid_unit", type=int, default=128)

    parser.add_argument("--expert_layers", type=int, default=1)
    parser.add_argument("--expert_out", type=int, default=512*3//2)
    parser.add_argument("--expert_hidden", type=int, default=512*3//2)
    parser.add_argument("--ple_layers", type=int, default=1)
    parser.add_argument("--num_experts", type=int, default=3)

    parser.add_argument("--num_tasks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tower_dropout", type=float, nargs="+")

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--use_esm2", action="store_true", default=False)
    parser.add_argument("--use_molebert", action="store_true", default=False)
    parser.add_argument("--use_add", action="store_true", default=False)
    parser.add_argument("--use_aux", action="store_true", default=False)
    parser.add_argument("--use_ple", action="store_true", default=False)

    parser.add_argument("--frozen_ligand_enc", action="store_true", default=False)
    parser.add_argument("--frozen_prot_enc", action="store_true", default=False)

    args = parser.parse_args()
    print(args)

    main(args)


if __name__ == "__main__":
    main_cli()
