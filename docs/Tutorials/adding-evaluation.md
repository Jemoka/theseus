# Adding an Evaluation

An evaluation runs the model over a fixed dataset and returns a scalar score. The most common pattern is `RolloutEvaluation`: provide a prompt and expected answer, and the framework handles generation and scoring.

---

## Minimal evaluation

```python
# theseus/evaluation/datasets/my_eval.py
from typing import Any, Tuple

from datasets import load_dataset

from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.data.tokenizer import decode_chat_template, encode_chat_template, get_tokenizer
from theseus.evaluation.base import RolloutEvaluation
from theseus.registry import evaluation


@evaluation("my_eval")
class MyEval(RolloutEvaluation):
    """One-line description shown in `theseus jobs`."""

    def __init__(self) -> None:
        self.ds = load_dataset("org/my-eval-dataset", split="test")
        self.encoder = get_tokenizer()

    # ── Required ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "my_eval"          # prefix for logged metrics: my_eval/score etc.

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx: int) -> Tuple[str, str]:
        """Return (prompt_string, expected_answer_string) for one example."""
        item = self.ds[indx]
        prompt = encode_chat_template(
            [ChatTurn(role="user", message=item["question"])],
            self.encoder,
            prompt=True,
            tokenize=False,
        )
        return prompt, item["answer"]

    def clean(self, y_hat: str) -> str:
        """Extract the model's answer from its raw generation."""
        chats: ChatTemplate = decode_chat_template(y_hat)
        for turn in chats:
            if turn.role == "assistant":
                return turn.message.strip()
        return ""
```

Register it:

```python
# theseus/evaluation/__init__.py  — add one line
from .datasets.my_eval import MyEval  # noqa: F401
```

---

## How it fits together

`RolloutEvaluation` calls `get(i)` to fetch a prompt, sends it to the model for generation, then calls `clean()` on the raw output before passing both the cleaned output and the expected answer to `check()`. `score()` averages `check()` over the whole dataset.

```
get(i) → (prompt, expected)
                │
         model generates
                │
         clean(raw_output) → y_hat
                │
         check(expected, y_hat) → bool
                │
         score() = mean(check results)
```

---

## Customising matching

Override `check()` for anything beyond exact string equality:

```python
def check(self, y: str, y_hat: str) -> bool:
    # Case-insensitive first-letter match (e.g. multiple choice A/B/C/D)
    return y.strip().upper()[:1] == y_hat.strip().upper()[:1]
```

---

## Controlling generation length

Override `max_new_tokens()` if the default is too short or too long for your task:

```python
def max_new_tokens(self, inference: Any) -> int:
    return 256    # default is typically 128
```

---

## Using your evaluation in an experiment

Add the evaluation key to the `eval.evaluations` list in your config YAML:

```yaml
eval:
  evaluations:
    - my_eval
    - mmlu
```

Results are logged to W&B under `my_eval/score` at each validation step.
