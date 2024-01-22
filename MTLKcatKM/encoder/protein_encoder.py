import gc

import torch
from torch import nn
from transformers import T5EncoderModel


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
