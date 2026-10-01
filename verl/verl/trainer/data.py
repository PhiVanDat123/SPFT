"""Single-turn SFT datasets, matching upstream DFT's prompt/response boundary."""

import torch
from torch.utils.data import Dataset, Sampler


class PromptResponseDataset(Dataset):
    """Parquet dataset with extra_info.question and extra_info.answer."""

    def __init__(self, path, tokenizer, max_length):
        import pyarrow.parquet as pq

        if max_length < 2:
            raise ValueError("max_length must be >= 2")
        self.rows = pq.read_table(path, columns=["extra_info"]).column(0).to_pylist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        if not self.rows:
            raise ValueError(f"Empty dataset: {path}")

    def __len__(self):
        return len(self.rows)

    def row_text(self, row):
        return row["question"], row["answer"]

    def __getitem__(self, index):
        # Validation pads shards with a real forward but zero metric contribution.
        dummy = index == -1
        row = self.rows[0 if dummy else index]
        question, answer = self.row_text(row)
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(
            answer + self.tokenizer.eos_token, add_special_tokens=False,
        )
        ids = (prompt_ids + answer_ids)[:self.max_length]
        length = len(ids)
        attention = torch.arange(self.max_length) < length
        # Mask is aligned with logits, predicting input_ids[t+1].
        positions = torch.arange(self.max_length - 1)
        mask = (positions >= len(prompt_ids) - 1) & (positions < length - 1)
        if dummy:
            mask.zero_()
        ids += [self.tokenizer.pad_token_id] * (self.max_length - length)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": attention.long(),
            "position_ids": (attention.long().cumsum(0) - 1).clamp_min(0),
            "loss_mask": mask,
        }


class NuminaDataset(PromptResponseDataset):
    pass


class UltraFeedbackDataset(PromptResponseDataset):
    pass


DATASETS = {
    "numina": NuminaDataset,
    "ultrafeedback": UltraFeedbackDataset,
}


def build_dataset(dataset_type, path, tokenizer, max_length):
    try:
        dataset_cls = DATASETS[dataset_type]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset_type {dataset_type!r}; choose {sorted(DATASETS)}") from exc
    return dataset_cls(path, tokenizer, max_length)


class ValidationSampler(Sampler):
    """Every real example exactly once; equal forward counts on all ranks."""
    def __init__(self, size, rank, world_size):
        self.size, self.rank, self.world_size = size, rank, world_size

    def __len__(self):
        return (self.size + self.world_size - 1) // self.world_size

    def __iter__(self):
        for offset in range(len(self)):
            index = offset * self.world_size + self.rank
            yield index if index < self.size else -1
