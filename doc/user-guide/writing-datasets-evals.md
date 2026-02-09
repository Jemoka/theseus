# Writing Datasets and Evaluations

This guide covers exactly how to extend:

- the dataset layer (two dataset pipelines), and
- the evaluation layer (three evaluation styles).

## Dataset Extension Model

There are two practical dataset pipelines in theseus.

## 1. Indexed datasets (for blockwise tokenization)

Use this path with `data/tokenize_blockwise_dataset`.

Inherit from one of:

- `StringDataset` for plain text items,
- `ChatTemplateDataset` for chat-structured items.

Both are aliases of `Dataset[T]` and require:

- `__len__(self) -> int`
- `__getitem__(self, idx: int) -> T`

### String dataset skeleton

```python
from theseus.data.datasets import StringDataset


class MyTextDataset(StringDataset):
    def __init__(self, split: str = "train") -> None:
        self.data = ["hello", "world"]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        return self.data[idx]
```

### Chat dataset skeleton

```python
from theseus.data.datasets import ChatTemplateDataset, ChatTurn


class MyChatDataset(ChatTemplateDataset):
    def __init__(self, split: str = "train") -> None:
        self.data = [
            [ChatTurn(role="user", message="Classify: great movie")],
            [ChatTurn(role="user", message="Classify: terrible film")],
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
```

## 2. Streaming datasets (for variable/pretraining tokenization)

Use this path with `data/tokenize_variable_dataset`.

Inherit from one of:

- `StreamingStringDataset`
- `StreamingChatTemplateDataset`

Both require:

- `__iter__(self) -> Iterator[T]`

For pretraining-style text streams, `StreamingPretrainingDataset` is the semantic alias typically used.

### Streaming pretraining skeleton

```python
from collections.abc import Iterator
from theseus.data.datasets import StreamingPretrainingDataset


class MyStream(StreamingPretrainingDataset):
    def __iter__(self) -> Iterator[str]:
        while True:
            yield "some training text"
```

## Registration

Register your dataset in `theseus/data/datasets/registry.py`:

```python
DATASETS["my_dataset"] = MyTextDataset
```

Then you can reference it via config key `data/dataset=my_dataset`.

## Evaluation Extension Model

All evaluations inherit from `Evaluation`, but you will almost always choose one of three concrete styles.

## 1. `RolloutEvaluation` (generation tasks)

Best for QA, classification via generated label, long-form prompting.

Required:

- `name` property
- `__len__`
- `get(self, idx) -> tuple[str, str]`  
  returns `(prompt, expected_answer)`
- `clean(self, y_hat: str) -> str`

Optional overrides:

- `check(self, y, y_hat) -> bool`
- `score(self, ys, y_hats) -> float`
- `max_new_tokens(self, inference) -> int`

### Skeleton

```python
from theseus.evaluation.base import RolloutEvaluation


class MyRolloutEval(RolloutEvaluation):
    @property
    def name(self) -> str:
        return "my_rollout"

    def __len__(self) -> int:
        return 100

    def get(self, idx: int) -> tuple[str, str]:
        return "Question?", "answer"

    def clean(self, y_hat: str) -> str:
        return y_hat.strip().lower()
```

## 2. `EncodingEvaluation` (next-token prediction style)

Best when correctness can be inferred from argmax continuation behavior.

Required:

- `name` property
- `__len__`
- `get(self, idx) -> str`  
  returns input string
- `clean(self, y_hat: str) -> str`

Optional:

- `check(self, x, y_hat) -> bool`
- `score(self, xs, y_hats) -> float`

### Skeleton

```python
from theseus.evaluation.base import EncodingEvaluation


class MyEncodingEval(EncodingEvaluation):
    @property
    def name(self) -> str:
        return "my_encoding"

    def __len__(self) -> int:
        return 100

    def get(self, idx: int) -> str:
        return "Input text"

    def clean(self, y_hat: str) -> str:
        return y_hat.strip()
```

## 3. `PerplexityEvaluation` (multiple-choice via likelihood)

Best for minimal-pair or MC settings where lower perplexity should identify the correct option.

Required:

- `name` property
- `__len__`
- `get(self, idx) -> tuple[str, list[str], int]`  
  returns `(prefix, continuations, correct_index)`

### Skeleton

```python
from theseus.evaluation.base import PerplexityEvaluation


class MyPplEval(PerplexityEvaluation):
    @property
    def name(self) -> str:
        return "my_ppl"

    def __len__(self) -> int:
        return 100

    def get(self, idx: int) -> tuple[str, list[str], int]:
        return "Prompt: ", ["good", "bad"], 0
```

## Registering Evaluations

Register in `theseus/evaluation/datasets/registry.py`:

```python
DATASETS["my_eval"] = MyRolloutEval
```

Enable in config:

```yaml
eval:
  evaluations:
    - my_eval
```

`Evaluator.from_trainer(...)` will instantiate registered evaluation classes by these names.

## Practical Tips

- Keep prompts and cleaning deterministic.
- Use small eval subsets while debugging correctness.
- For rollout evals, cap `max_new_tokens` aggressively.
- Reuse tokenizer conventions from training (HF `apply_chat_template` vs ChatML).
