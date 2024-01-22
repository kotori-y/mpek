import argparse
import gc
import json
import os
import pickle
from collections import defaultdict

import pandas as pd
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
from MTLKcatKM.encoder import ProteinEncoder, AuxiliaryEncoder, LigandEncoderMoleBert
from MTLKcatKM.model import MTLModel
from MTLKcatKM.utils import exempt_parameters, get_device, init_distributed_mode, TrainNormalizer, show_me_grad, \
    WarmCosine, ChildTuningAdamW

import warnings
warnings.filterwarnings("ignore")


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)  # sum-up as the all-reduce operation
    rt /= nprocs  # NOTE this is necessary, since all_reduce here do not perform average
    return rt


def evaluate(model: MTLModel, loader, device, normalizer: TrainNormalizer, args):
    model.eval()

    pred_result = {}

    pbar = tqdm(loader, desc="Iteration", disable=False)

    for step, batch in enumerate(pbar):
        net_input = {
            "mol_graph": batch["mol_graph"].to(device),
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "organ_ids": None,
            "condition": None
            # "condition": None
        }

        if args.use_organism:
            net_input["organ_ids"] = batch["organ_ids"].to(device)
        if args.use_ph or args.use_temperature:
            net_input["condition"] = {k: v.to(device) for k, v in batch["condition"].items()}

        with torch.no_grad():
            labels = {k: v.to(device) for k, v in batch["labels"].items()}
            net_outputs = model(**net_input, task_names=list(labels.keys()))

            for task in list(labels.keys()):
                _y_pred = net_outputs[task].cpu().detach().numpy().flatten()

                pred_result[f"{task}_pred"] = np.hstack(
                    [pred_result.get(f"{task}_pred", []), _y_pred]
                )

    _out = list(pred_result.values())
    out = np.vstack(_out).T
    out = normalizer.denorm(out).T.tolist()

    return dict(zip(list(labels.keys()), out))


def main(args):
    device = get_device(args.device, args.local_rank)

    prot_enc = ProteinEncoder(model_dir=args.prottrans_path, device=device, frozen_params=True)
    lig_enc = LigandEncoderMoleBert(init_model=args.molebert_dir, device=device, frozen_params=True)

    with open(args.organism_dictionary_path) as f:
        _dict = json.load(f)

    aux_enc = AuxiliaryEncoder(
        organism_dict_size=len(_dict),
        embed_size=lig_enc.embed_dim,
        # embed_size=lig_enc.embed_dim + prot_enc.embed_dim,
        use_ph=args.use_ph,
        use_temperature=args.use_temperature,
        use_organism=args.use_organism,
        device=device
    )

    model = MTLModel(
        ligand_enc=lig_enc, protein_enc=prot_enc, auxiliary_enc=aux_enc,
        expert_out=args.expert_out, expert_hidden=args.expert_hidden,
        expert_layers=args.expert_layers, num_experts=args.num_experts,
        tower_hid_layer=args.tower_hid_layer, tower_hid_unit=args.tower_hid_unit,
        num_tasks=args.num_tasks, dropout=args.dropout, device=device,
        use_ph=args.use_ph, use_temperature=args.use_temperature, use_organism=args.use_organism,
        tower_dropout=args.tower_dropout, num_ple_layers=args.ple_layers
    ).to(device)
    gc.collect()

    if args.checkpoint_dir:

        params = torch.load(args.checkpoint_dir, map_location=device)

        model.load_state_dict(params["model_state_dict"])
        print("loaded params successful")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"#Params: {num_params}")

    dataset_params = {
        "organism_dictionary_path": args.organism_dictionary_path,
        # "sequence_index": args.sequence_index,
        "sequence_column": args.sequence_column,
        # "sequence_column_end": args.sequence_column_end,
        "smiles_column": args.smiles_column,
        "label_column": args.label_column,
        "max_length": args.max_length,
        "organism_column": args.organism_column,
        "ph_column": args.ph_column,
        "temperature_column": args.temperature_column,
        "use_ph": args.use_ph,
        "use_temperature": args.use_temperature,
        "use_organism": args.use_organism,
        "num_tasks": args.num_tasks
    }

    with open(args.train_normalizer_path, 'rb') as f:
        train_normalizer = pickle.load(f)

    test_dataset = MTLKDataset(
        data=args.test_data, data_path=args.test_path, model_name=args.prottrans_path,
        normalizer=train_normalizer, evaluate=True, **dataset_params
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("Evaluating...")
    test_pred = evaluate(model, test_loader, device=device, normalizer=train_normalizer, args=args)
    print(test_pred)
    # pred_df = pd.DataFrame(test_pred)
    # pred_df.to_csv(f'{args.result_file}', index=False)


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--checkpoint_dir", type=str, default=None)
    # parser.add_argument("--result_file", type=str, default="./pred.csv")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--test_path", type=str)

    parser.add_argument("--organism_dictionary_path", type=str)
    parser.add_argument("--train_normalizer_path", type=str)

    parser.add_argument("--sequence_column", type=str, default=0)
    parser.add_argument("--smiles_column", type=str, default=1)
    parser.add_argument("--organism_column", type=str)
    parser.add_argument("--ph_column", type=str)
    parser.add_argument("--temperature_column", type=str)
    parser.add_argument("--label_column", type=str, nargs="+", default=None)

    parser.add_argument("--max_length", type=int, default=4)

    parser.add_argument("--prottrans_path", type=str)
    parser.add_argument("--molebert_dir", type=str, default=None)

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

    parser.add_argument("--use_ph", action="store_true", default=False)
    parser.add_argument("--use_temperature", action="store_true", default=False)
    parser.add_argument("--use_organism", action="store_true", default=False)

    args = parser.parse_args()
    print(args)

    main(args)


if __name__ == "__main__":
    main_cli()
