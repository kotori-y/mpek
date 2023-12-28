import gc
import os.path
import warnings
import re

import numpy as np
import torch
from torch import nn
from transformers import T5EncoderModel
# from PyBioMed.PyProtein import AAComposition

import esm
from esm.model.esm2 import ESM2


class ProteinEncoder(nn.Module):
    def __init__(self, model_dir, device=None, frozen_params=False):
        super().__init__()
        self.embed_dim = 1024
        self.frozen_params = frozen_params

        self.device = device
        if device is None:
            self.device = torch.device('cpu')

        self.encoder = T5EncoderModel.from_pretrained(model_dir)
        if frozen_params:
            print(f"frozen {self}")
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            for name, p in self.encoder.named_parameters():
                if "encoder.lm_head" not in name:
                    p.requires_grad = False
            print(f"finetune {self}")

        gc.collect()

    def forward(self, input_ids, attention_mask, shift_left=0, shift_right=-1):
        if self.frozen_params:
            self.encoder.eval()
            with torch.no_grad():
                embeddings = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = embeddings.last_hidden_state

                features = []
                for i, embedding in enumerate(embeddings):
                    seq_len = (attention_mask[i] == 1).sum()
                    seq_emd = embedding[:seq_len - 1]
                    features.append(seq_emd.mean(axis=0))
                return torch.vstack(features)

        embeddings = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = embeddings.last_hidden_state

        features = []
        for i, embedding in enumerate(embeddings):
            seq_len = (attention_mask[i] == 1).sum()
            seq_emd = embedding[:seq_len - 1]
            features.append(seq_emd.mean(axis=0))
        return torch.vstack(features)

    def __str__(self):
        return "Protein Encoder"


class ProteinEncoderESM2(nn.Module):

    def __init__(self, esm_dir, device=None, frozen_params=False):
        super().__init__()
        self.embed_dim = 1280
        self.frozen_params = frozen_params

        self.device = device
        if device is None:
            self.device = torch.device('cpu')

        model_path = os.path.join(esm_dir, 'esm2_t33_650M_UR50D.pt')
        regr_path = os.path.join(esm_dir, 'esm2_t33_650M_UR50D-contact-regression.pt')

        model_data = torch.load(model_path, map_location=device)
        regr_data = torch.load(regr_path, map_location=device)

        self.encoder, self.alphabet = ProteinEncoderESM2.load_model_and_alphabet_core(model_data, regr_data)
        print(self.alphabet)

        if self.frozen_params:
            print(f"frozen {self}")
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            for name, p in self.encoder.named_parameters():
                if "encoder.lm_head" not in name:
                    p.requires_grad = False
            print(f"finetune {self}")

    def forward(self, input_ids):
        if self.frozen_params:
            self.encoder.eval()
            with torch.no_grad():
                return self.encoder(input_ids, repr_layers=[33], return_contacts=True)['representations'][33][:, 0, :]

        return self.encoder(input_ids, repr_layers=[33], return_contacts=True)['representations'][33][:, 0, :]

    def __str__(self):
        return "ESM2 Protein Encoder"

    def run_demo(self):
        self.encoder.eval()  # disables dropout for deterministic results

        # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
        data = [
            ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
            ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
            ("protein2 with mask", "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
            ("protein3", "K A <mask> I S Q"),
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        print(self.alphabet.encode("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"))

    @staticmethod
    def _load_model_and_alphabet_core_v2(model_data):
        def upgrade_state_dict(state_dict):
            """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
            prefixes = ["encoder.sentence_encoder.", "encoder."]
            pattern = re.compile("^" + "|".join(prefixes))
            state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
            return state_dict

        cfg = model_data["cfg"]["model"]
        state_dict = model_data["model"]
        state_dict = upgrade_state_dict(state_dict)
        alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        model = ESM2(
            num_layers=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,
            attention_heads=cfg.encoder_attention_heads,
            alphabet=alphabet,
            token_dropout=cfg.token_dropout,
        )
        return model, alphabet, state_dict

    @staticmethod
    def load_model_and_alphabet_core(model_data, regression_data=None):
        if regression_data is not None:
            model_data["model"].update(regression_data["model"])

        model, alphabet, model_state = ProteinEncoderESM2._load_model_and_alphabet_core_v2(model_data)
        expected_keys = set(model.state_dict().keys())
        found_keys = set(model_state.keys())

        if regression_data is None:
            expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
            error_msgs = []
            missing = (expected_keys - found_keys) - expected_missing
            if missing:
                error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
            unexpected = found_keys - expected_keys
            if unexpected:
                error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

            if error_msgs:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        model.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )
            if expected_missing - found_keys:
                warnings.warn(
                    "Regression weights not found, predicting contacts will not produce correct results."
                )

        model.load_state_dict(model_state, strict=regression_data is not None)

        return model, alphabet

# class ProteinEncoder:
#     def __init__(self, device=None):
#         super().__init__()
#         self.device = device
#         self.embed_dim = 9790
#
#         if device is None:
#             self.device = torch.device('cpu')
#
#     def __call__(self, sequence):
#         # arr = []
#         #
#         # for seq in sequence:
#         #     vec = list(AAComposition.CalculateAAComposition(seq).values())
#         #     arr.append(vec)
#         #
#         # matrix = np.array(arr).astype("float32")
#         # sequence = sequence.astype("float32")
#         return sequence.to(self.device)


if __name__ == "__main__":
    # from MTLKcatKM.datasets import MTLKDataloader
    #
    model_name = "/fs1/home/wangll/software/MTLKcatKM/pretrained/prot_t5_xl_uniref50"
    #
    # dataset = MTLKDataloader(
    #     batch_size=2, num_workers=40,
    #     test_size=0.16, valid_size=0.16,
    #     data_path="../datasets/total.csv", model_name=model_name,
    #     max_length=128
    # )
    #
    enc = ProteinEncoder(model_dir=model_name, frozen_params=True)
    # _train_loader, _, _ = dataset.get_data_loaders()
    # for i, batch in enumerate(_train_loader):
    #     h = enc(batch["input_ids"], batch["attention_mask"])
    #     print(h, h.shape)
    #     break

    # enc = ProteinEncoderESM2('/fs1/home/wangll/software/ESMFold/esm2')
    n = 0
    for name, _ in enc.named_parameters():
        print(name)
        n += 1
    print(n)
