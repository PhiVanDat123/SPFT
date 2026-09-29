import torch
from verl.trainer.data import NuminaDataset, ValidationSampler


class Tokenizer:
    pad_token_id = 0
    eos_token = " EOS"

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def encode(self, text, **kwargs):
        return list(range(1, len(text.split()) + 1))


def make_dataset(tmp_path, max_length=8):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "tiny.parquet"
    pq.write_table(pa.Table.from_pylist([{"extra_info": {"question": "a b c", "answer": "d e"}}]), path)
    return NuminaDataset(path, Tokenizer(), max_length)


def test_answer_mask_and_dummy(tmp_path):
    ds = make_dataset(tmp_path)
    assert ds[0]["loss_mask"].tolist() == [False, False, True, True, True, False, False]
    assert ds[0]["attention_mask"].sum().item() == 6
    assert not ds[-1]["loss_mask"].any()
    assert torch.equal(ds[0]["input_ids"], ds[-1]["input_ids"])


def test_truncated_prompt_has_no_answer(tmp_path):
    assert not make_dataset(tmp_path, max_length=3)[0]["loss_mask"].any()


def test_validation_shards_no_duplicates():
    for size in (1, 3, 5, 10):
        for world in (1, 2, 4):
            shards = [list(ValidationSampler(size, rank, world)) for rank in range(world)]
            assert len({len(shard) for shard in shards}) == 1
            assert sorted(i for shard in shards for i in shard if i >= 0) == list(range(size))
