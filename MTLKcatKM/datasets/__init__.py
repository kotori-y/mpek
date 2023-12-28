import os
import re
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append("../..")
from MTLKcatKM.utils import TrainNormalizer

from transformers import T5Tokenizer

try:
    from dataset_test import MolTestDataset
except:
    from .dataset_test import MolTestDataset

from Mole_BERT.loader import mol_to_graph_data_obj_simple

from MTLKcatKM.utils import OrganismTokenizer
from esm.data import Alphabet

from rdkit import Chem


def read_data(
        data_path, sequence_column, smiles_column, num_tasks=None,
        label_column=None, organism_column=None, ph_column=None, temperature_column=None,
        eval=False, only_norm=False, use_ph=True, use_temperature=True, use_organism=True,
):
    _, ext = os.path.splitext(data_path)
    sep = "," if ext == ".csv" else "\t"

    data = pd.read_csv(data_path, sep=sep)
    # data = _data.values

    sequence_data = data.loc[:, sequence_column].values
    smiles_data = data.loc[:, smiles_column].values

    organisms = data.loc[:, organism_column].values if use_organism else None
    phs = data.loc[:, ph_column].values.astype(np.float32) if use_ph else None
    temperatures = data.loc[:, temperature_column].values.astype(np.float32) if use_temperature else None

    if eval:
        labels = np.random.random((len(data), num_tasks))
        task_names = [f'task_{n}' for n in range(num_tasks)]
    else:
        labels = data.loc[:, label_column].values.astype(np.float32)
        task_names = label_column

    return sequence_data, smiles_data, organisms, phs, temperatures, labels, task_names


class MTLKDataset(MolTestDataset):
    def __init__(
            self, data_path, organism_dictionary_path,
            sequence_column, smiles_column,
            label_column, organism_column,
            ph_column, temperature_column,
            model_name, max_length=256,
            normalizer: TrainNormalizer = None, evaluate=False,
            use_ph=True, use_temperature=True, use_organism=True,
            num_tasks=2
    ):
        super(MolTestDataset, self).__init__(data_path)
        self.max_length = max_length

        self.sequence_data, self.smiles_data, self.organisms, self.phs, self.temperatures, self.labels, self.task_names = \
            read_data(
                data_path,
                smiles_column=smiles_column, sequence_column=sequence_column, ph_column=ph_column,
                temperature_column=temperature_column, organism_column=organism_column, label_column=label_column,
                eval=evaluate, use_ph=use_ph, use_temperature=use_temperature, use_organism=use_organism,
                num_tasks=num_tasks
            )

        # if self.use_esm2:
        #     seq_encoded_list = [alphabet.encode(seq_str) for seq_str in self.sequence_data]
        #     seq_encoded_list = [seq_str[:self.max_length] for seq_str in seq_encoded_list]
        #
        #     self.tokens = torch.empty(
        #         (
        #             len(seq_encoded_list),
        #             max_length + int(alphabet.prepend_bos) + int(alphabet.append_eos),
        #         ),
        #         dtype=torch.int64,
        #     )
        #
        #     self.tokens.fill_(alphabet.padding_idx)
        #     for i, seq_encoded in enumerate(seq_encoded_list):
        #         if alphabet.prepend_bos:
        #             self.tokens[i, 0] = alphabet.cls_idx
        #         seq = torch.tensor(seq_encoded, dtype=torch.int64)
        #         self.tokens[
        #         i,
        #         int(alphabet.prepend_bos): len(seq_encoded)
        #                                    + int(alphabet.prepend_bos),
        #         ] = seq
        #         if alphabet.append_eos:
        #             self.tokens[i, len(seq_encoded) + int(alphabet.prepend_bos)] = alphabet.eos_idx
        # else:
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)

        self.organism_tokenizer = OrganismTokenizer(organism_dictionary_path)

        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = TrainNormalizer(train_path=data_path, label_index=label_column)

        # self.labels = np.log2(10 ** self.labels)
        self.labels = self.normalizer.norm(self.labels).astype(np.float32)
        # self.labels = self.labels.astype(np.float32)
        # self.sequence_data = self.normalizer.norm_x(self.sequence_data).astype(np.float32)

        self.task = 'regression'

        self.use_ph = use_ph
        self.use_temperature = use_temperature
        self.use_organism = use_organism

    def __getitem__(self, item):
        data, y = super().__getitem__(item)

        smiles = self.smiles_data[item]
        data = mol_to_graph_data_obj_simple(Chem.MolFromSmiles(smiles))

        seq = " ".join(list(self.sequence_data[item]))
        seq = re.sub(r"[UZOB]", "X", seq)
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)
        # seq_ids = self.tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}

        organ_ids = torch.tensor(self.organism_tokenizer.tokenize(self.organisms[item]),
                                 dtype=torch.int) if self.use_organism else None
        # condition = torch.tensor(self.conditions[item], dtype=torch.float32)
        sample["mol_graph"] = data

        if self.use_organism:
            sample["organ_ids"] = organ_ids

        if self.use_ph or self.use_temperature:
            sample["condition"] = {}

            if self.use_ph:
                pH = torch.tensor(self.phs[item], dtype=torch.float32)
                sample["condition"]["pH"] = pH

            if self.use_temperature:
                temperature = torch.tensor(self.temperatures[item], dtype=torch.float32)
                sample["condition"]["temperature"] = temperature

        sample["labels"] = dict(zip(self.task_names, y))
        return sample


if __name__ == "__main__":

    dataset_params = {
        "organism_dictionary_path": '../data/new_modeling/demo/organism_token.json',
        # "sequence_column": args.sequence_column,
        "sequence_column": 11,
        # "sequence_column_end": args.sequence_column_end,
        "smiles_column": 10,
        "label_column": [14, 15],
        "max_length": 32,
        "organism_column": 2,
        "ph_column": 6,
        "temperature_column": 7,
        "use_esm2": True,
        "use_molebert": True
    }

    from MTLKcatKM.encoder import ProteinEncoderESM2

    demo = ProteinEncoderESM2('/fs1/home/wangll/software/ESMFold/esm2')

    train_path = "../data/fin/fin/all_XOBU/test_all_XOBU.txt"
    train_dataset = MTLKDataset(data_path=train_path, model_name="../pretrained/prot_t5_xl_uniref50",
                                alphabet=demo.alphabet, **dataset_params)

    for batch in train_dataset:
        print(batch["mol_graph"].x)
        break
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=16,
    #     num_workers=10,
    #     shuffle=False
    # )
    #
    # for batch in train_loader:
    #     print(demo(batch['input_ids']).shape)
    #     break
