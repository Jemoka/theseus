from dataclasses import dataclass
from typing import Union, List, Any

from theseus.data.datasets import Dataset, StreamingDataset


@dataclass
class RecipeEntry:
    dataset: Union[Dataset[Any], StreamingDataset[Any]]
    split: float


Recipe = List[RecipeEntry]


# from theseus.base import local, ExecutionSpec

# spec = ExecutionSpec(
#     name="tokenize",
#     hardware=local("/Users/houjun/theseus/theseus-prod-fs", "/Users/houjun/Worktrees/"),
#     distributed=False,
# )
