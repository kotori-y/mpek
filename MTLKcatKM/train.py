import argparse
import gc
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DistributedSampler, RandomSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.append("../")
from MTLKcatKM.datasets import MTLKDataset
from MTLKcatKM.encoder import ProteinEncoder, AuxiliaryEncoder, LigandEncoderMoleBert

# from MTLKcatKM.encoder import ProteinEncoderESM2, LigandEncoder

from MTLKcatKM.model import MTLModel
from MTLKcatKM.utils import exempt_parameters, get_device, init_distributed_mode, TrainNormalizer, \
    WarmCosine, ChildTuningAdamW

import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score, mean_squared_error


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)  # sum-up as the all-reduce operation
    rt /= nprocs  # NOTE this is necessary, since all_reduce here do not perform average
    return rt


def train(model: MTLModel, loader, optimizer, device, args, scheduler):
    model.train()

    pred_history = {}

    loss_accum_dict = defaultdict(float)
    r2_accum_dict = defaultdict(float)

    pbar = tqdm(loader, desc="Iteration", disable=args.disable_tqdm)

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

        labels = {k: v.to(device) for k, v in batch["labels"].items()}
        net_outputs = model(**net_input, task_names=list(labels.keys()))

        optimizer.zero_grad()

        if args.distributed:
            loss, loss_dict = model.module.compute_loss(net_outputs, labels)
        else:
            loss, loss_dict = model.compute_loss(net_outputs, labels)

        # if args.rank != 0:
        # show_me_grad(model)

        loss.backward()
        optimizer.step()
        scheduler.step()

        for k, v in loss_dict.items():
            loss_accum_dict[k] += v

        # i = 0
        # for task_name, target in labels.items():
        #     y_true = target.cpu().numpy()
        #     y_pred = net_outputs[i].detach().cpu().numpy().flatten()
        #     i += 1
        #     r2_accum_dict[task_name] += r2_score(y_true, y_pred)

        if step % args.log_interval == 0:
            description = f"Iteration loss: {loss_accum_dict['loss'] / (step + 1):6.4f}"
            # for task_name in labels.keys():
            #     description += f"Iteration {task_name} r2: {loss_accum_dict[task_name] / (step + 1):6.4f}"
            description += f" lr: {scheduler.get_last_lr()[0]:.5e}"
            pbar.set_description(description)

        for task in list(labels.keys()):
            _y_true = labels[task].cpu().detach().numpy().flatten()
            _y_pred = net_outputs[task].cpu().detach().numpy().flatten()

            mask = np.isnan(_y_true)

            y_true = _y_true[~mask]
            y_pred = _y_pred[~mask]

            pred_history[f"{task}_true"] = np.hstack(
                [pred_history.get(f"{task}_true", []), y_true]
            )
            pred_history[f"{task}_pred"] = np.hstack(
                [pred_history.get(f"{task}_pred", []), y_pred]
            )

    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= (step + 1)
    # for k in r2_accum_dict.keys():
    #     r2_accum_dict[k] /= (step + 1)

    for task in list(labels.keys()):
        task_true = pred_history[f"{task}_true"]
        task_pred = pred_history[f"{task}_pred"]

        loss_accum_dict[f'{task}_r2'] = r2_score(task_true, task_pred)
        loss_accum_dict[f'{task}_rmse'] = mean_squared_error(task_true, task_pred) ** 0.5

    # for task in list(labels.keys()):
    #     loss_accum_dict[f'{task}_r2'] = r2_score(pred_history[f"{task}_true"], pred_history[f"{task}_pred"])

    return {**loss_accum_dict}, pred_history


def evaluate(model: MTLModel, loader, device, normalizer: TrainNormalizer, args):
    model.eval()

    loss_accum_dict = defaultdict(float)

    pred_history = {}

    pbar = tqdm(loader, desc="Iteration", disable=args.disable_tqdm)

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
                _y_true = labels[task].cpu().detach().numpy().flatten()
                _y_pred = net_outputs[task].cpu().detach().numpy().flatten()

                mask = np.isnan(_y_true)

                y_true = _y_true[~mask]
                y_pred = _y_pred[~mask]

                pred_history[f"{task}_true"] = np.hstack(
                    [pred_history.get(f"{task}_true", []), y_true]
                )
                pred_history[f"{task}_pred"] = np.hstack(
                    [pred_history.get(f"{task}_pred", []), y_pred]
                )

            # print(pred_history)
            if args.distributed:
                loss, loss_dict = model.module.compute_loss(net_outputs, labels)
                # r2_dict = model.module.compute_r2(net_outputs, labels)
            else:
                loss, loss_dict = model.compute_loss(net_outputs, labels)
                # r2_dict = model.compute_r2(net_outputs, labels)

            for k, v in loss_dict.items():
                loss_accum_dict[k] += v

            # for k, v in r2_dict.items():
            #     r2_accum_dict[k] += v

            # for task_name, target in labels.items():
            #     y_pred = net_outputs[task_name].flatten()
            #     r2_accum_dict[task_name] += r2_score(y_pred, target)

    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= step + 1
    # for k in r2_accum_dict.keys():
    #     r2_accum_dict[k] /= (step + 1)

    for task in list(labels.keys()):
        task_true = pred_history[f"{task}_true"]
        task_pred = pred_history[f"{task}_pred"]

        loss_accum_dict[f'{task}_r2'] = r2_score(task_true, task_pred)
        loss_accum_dict[f'{task}_rmse'] = mean_squared_error(task_true, task_pred) ** 0.5

    return loss_accum_dict, pred_history


def main(args):
    device = get_device(args.device, args.local_rank)
    # device = get_device(args.device)

    prot_enc = ProteinEncoder(model_dir=args.prottrans_path, device=device, frozen_params=args.frozen_prot_enc)
    lig_enc = LigandEncoderMoleBert(init_model=args.molebert_dir, device=device, frozen_params=args.frozen_ligand_enc)

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
        weights=args.weights, tower_dropout=args.tower_dropout, num_ple_layers=args.ple_layers
    ).to(device)
    gc.collect()

    model_without_ddp = model
    args.disable_tqdm = False

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
        )
        model_without_ddp = model.module
        args.disable_tqdm = args.rank != 0

    num_params = sum(p.numel() for p in model_without_ddp.parameters())
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
    }

    train_dataset = MTLKDataset(data_path=args.train_path, model_name=args.prottrans_path, **dataset_params)
    valid_dataset = MTLKDataset(data_path=args.valid_path, model_name=args.prottrans_path,
                                normalizer=train_dataset.normalizer, **dataset_params)
    test_dataset = MTLKDataset(data_path=args.test_path, model_name=args.prottrans_path,
                               normalizer=train_dataset.normalizer, **dataset_params)

    if args.distributed:
        sampler_train = DistributedSampler(train_dataset)
    else:
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

    ligand_encoder_params = lig_enc.parameters()
    protein_encoder_params = prot_enc.parameters()
    head_params = exempt_parameters(
        model_without_ddp.parameters(),
        list(ligand_encoder_params) + list(protein_encoder_params)
    )
    # head_params = exempt_parameters(model_without_ddp.parameters(), [])
    #
    params = [{'params': head_params, 'lr': args.head_lr}]

    if not args.frozen_ligand_enc:
        params.append({'params': ligand_encoder_params, 'lr': args.ligand_enc_lr})
    if not args.frozen_prot_enc:
        params.append({'params': protein_encoder_params, 'lr': args.protein_enc_lr})
    # optimizer = torch.optim.Adam(
    #     params,
    #     betas=(0.9, args.beta2),
    #     weight_decay=args.weight_decay
    # )

    optimizer = ChildTuningAdamW(
        params,
        betas=(0.9, args.beta2),
        weight_decay=args.weight_decay
    )

    if not args.lr_warmup:
        scheduler = LambdaLR(optimizer, lambda x: 1.0)
    else:
        lrscheduler = WarmCosine(tmax=len(train_loader) * args.period, warmup=int(4e3))
        scheduler = LambdaLR(optimizer, lambda x: lrscheduler.step(x))

    train_curve = []
    valid_curve = []
    test_curve = []

    train_pred_history = []
    valid_pred_history = []
    test_pred_history = []

    for epoch in range(1, args.epochs + 1):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        print("=====Epoch {}".format(epoch))

        print("Training...")
        train_dict, _train_pred_history = train(
            model, train_loader, device=device, args=args, optimizer=optimizer, scheduler=scheduler
        )

        print("Evaluating...")
        valid_dict, _valid_pred_history = evaluate(model, valid_loader, device=device,
                                                   normalizer=train_dataset.normalizer, args=args)
        test_dict, _test_pred_history = evaluate(model, test_loader, device=device, normalizer=train_dataset.normalizer,
                                                 args=args)

        _train_pred_history["epoch"] = _valid_pred_history["epoch"] = _test_pred_history["epoch"] = epoch

        print(
            f"train loss: {train_dict} \n "
            f"valid loss: {valid_dict} \n "
            f"test loss: {test_dict}"
        )

        train_curve.append(train_dict)
        valid_curve.append(valid_dict)
        test_curve.append(test_dict)

        train_pred_history.append(_train_pred_history)
        valid_pred_history.append(_valid_pred_history)
        test_pred_history.append(_test_pred_history)

        rank = dist.get_rank() if args.distributed else 0

        out_dir = args.out_dir

        for pref, curve, pred in zip(
                ["train", "valid", "test"],
                [train_curve, valid_curve, test_curve],
                [train_pred_history, valid_pred_history, test_pred_history]
        ):
            # for pref, curve in zip(["train"], [train_curve]):
            if not os.path.exists(f'./{out_dir}/{pref}_curve'):
                os.makedirs(f'./{out_dir}/{pref}_curve')

            root = f'./{out_dir}/{pref}_curve'

            with open(f"./{root}/{pref}_history_{rank}.pkl", 'wb') as f:
                pickle.dump(curve, f)

            with open(f"./{root}/{pref}_pred_{rank}.pkl", 'wb') as f2:
                pickle.dump(pred, f2)

        if rank == 0 and epoch >= 25:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model_without_ddp.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": args,
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"checkpoint_{epoch}.pt"))

    if args.distributed:
        torch.distributed.destroy_process_group()


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
    parser.add_argument("--sequence_column", type=int, default=0)
    # parser.add_argument("--sequence_index_end", type=int, default=0)
    parser.add_argument("--smiles_column", type=int, default=1)
    parser.add_argument("--organism_column", type=int)
    parser.add_argument("--ph_column", type=int)
    parser.add_argument("--temperature_column", type=int)
    parser.add_argument("--label_column", type=int, nargs="+")
    parser.add_argument("--weights", type=float, nargs="+")

    parser.add_argument("--max_length", type=int, default=4)

    # parser.add_argument("--molclr_path", type=str)
    parser.add_argument("--prottrans_path", type=str)
    # parser.add_argument("--esm_dir", type=str, default=None)
    parser.add_argument("--molebert_dir", type=str, default=None)

    parser.add_argument("--tower_hid_layer", type=int, default=1)
    parser.add_argument("--tower_hid_unit", type=int, default=128)

    parser.add_argument("--expert_layers", type=int, default=1)
    parser.add_argument("--expert_out", type=int, default=512 * 3 // 2)
    parser.add_argument("--expert_hidden", type=int, default=512 * 3 // 2)
    parser.add_argument("--ple_layers", type=int, default=1)
    parser.add_argument("--num_experts", type=int, default=3)

    parser.add_argument("--num_tasks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tower_dropout", type=float, nargs="+")

    parser.add_argument("--ligand_enc_lr", type=float, default=1e-5)
    parser.add_argument("--protein_enc_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-5)

    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--period", type=float, default=10)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--debug", action="store_true", default=False)

    parser.add_argument("--use_ph", action="store_true", default=False)
    parser.add_argument("--use_temperature", action="store_true", default=False)
    parser.add_argument("--use_organism", action="store_true", default=False)

    parser.add_argument("--frozen_ligand_enc", action="store_true", default=False)
    parser.add_argument("--frozen_prot_enc", action="store_true", default=False)

    args = parser.parse_args()
    if args.distributed:
        init_distributed_mode(args)
    print(args)

    main(args)


if __name__ == "__main__":
    main_cli()

    # parser.add_argument("--global-reducer", type=str, default="sum")
    # parser.add_argument("--node-reducer", type=str, default="sum")
    # parser.add_argument("--graph-pooling", type=str, default="sum")
    # parser.add_argument("--dropedge-rate", type=float, default=0.1)
    # parser.add_argument("--dropnode-rate", type=float, default=0.1)
    # parser.add_argument("--num-layers", type=int, default=6)
    # parser.add_argument("--decoder-layers", type=int, default=None)
    # parser.add_argument("--latent-size", type=int, default=256)
    # parser.add_argument("--mlp-hidden-size", type=int, default=1024)
    # parser.add_argument("--mlp_layers", type=int, default=2)
