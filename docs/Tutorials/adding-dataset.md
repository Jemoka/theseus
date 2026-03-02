# Adding a Dataset

A dataset yields text for pretraining or chat turns for fine-tuning. There are four types of datasets, each of which you can inherit per your needs.

## Dataset types

| Base class | Data shape | When to use |
|---|---|---|
| `StreamingPretrainingDataset` | yields `str` | Large pretraining corpora; tokenised irrespective of item boundaries |
| `PretrainingDataset` | `__getitem__` → `str` | Same semantics but finite and indexable |
| `ChatTemplateDataset` | `__getitem__` → `ChatTemplate` | Instruction / chat fine-tuning |
| `StreamingChatTemplateDataset` | yields `ChatTemplate` | Streaming chat data |

---

## Pretraining dataset (streaming)

This is the most common case for large-scale pretraining data from HuggingFace:

```python
# theseus/data/datasets/my_corpus.py
from collections.abc import Iterator

from datasets import load_dataset
from theseus.data.datasets import StreamingPretrainingDataset
from theseus.registry import dataset


@dataset("my_corpus")
class MyCorpus(StreamingPretrainingDataset):
    def __init__(self, config: str | None = None) -> None:
        self.ds = load_dataset("org/my-corpus", split="train", streaming=True)

    def __iter__(self) -> Iterator[str]:
        for item in self.ds:
            yield item["text"]
```

Register it:

```python
# theseus/data/datasets/__init__.py  — add one line
from .my_corpus import MyCorpus  # noqa: F401
```

---

## Chat / instruction dataset

For fine-tuning on instruction-following data, yield `ChatTemplate` — a list of `ChatTurn` objects:

```python
# theseus/data/datasets/my_chat.py
from datasets import load_dataset

from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn
from theseus.registry import dataset


@dataset("my_chat")
class MyChat(ChatTemplateDataset):
    def __init__(self, split: str = "train", config: str | None = None) -> None:
        self.ds = load_dataset("org/my-chat-dataset", split=split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        return [
            ChatTurn(role="user",      message=item["instruction"]),
            ChatTurn(role="assistant", message=item["response"]),
        ]
```

A `ChatTemplate` is just a `list[ChatTurn]`. Roles are free-form strings but `"user"`, `"assistant"`, and `"system"` are the conventional values.

---

## Dataset with multiple splits

Add a `split` argument to `__init__` and the caller can request `"train"` or `"validation"` when configuring data pipelines:

```python
@dataset("my_corpus")
class MyCorpus(StreamingPretrainingDataset):
    def __init__(self, split: str = "train", config: str | None = None) -> None:
        self.ds = load_dataset("org/my-corpus", split=split, streaming=True)

    def __iter__(self) -> Iterator[str]:
        for item in self.ds:
            yield item["text"]
```

---

## Using your dataset in an experiment

Once registered, reference the dataset key in your config YAML under `training.dataset`:

```yaml
training:
  dataset:
    - name: my_corpus
      weight: 1.0
```

Multiple datasets can be mixed by weight:

```yaml
training:
  dataset:
    - name: fineweb
      weight: 0.8
    - name: my_corpus
      weight: 0.2
```
