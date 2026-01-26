from typing import Generic, TypeVar, List, Dict, Iterator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

C = TypeVar("C")


class Dataset(ABC, Generic[C]):
    @abstractmethod
    def __getitem__(self, indx: int) -> C: ...

    @abstractmethod
    def __len__(self) -> int: ...


@dataclass
class ChatTurn:
    role: str
    message: str
    metadata: Dict[str, str] = field(default_factory=dict)


ChatTemplate = List[ChatTurn]

StringDataset = Dataset[str]
ChatTemplateDataset = Dataset[ChatTemplate]


class StreamingDataset(ABC, Generic[C]):
    @abstractmethod
    def __iter__(self) -> Iterator[C]: ...

    def __len__(self) -> int:
        raise NotImplementedError("This dataset does not support __len__")


StreamingStringDataset = StreamingDataset[str]
StreamingChatTemplateDataset = StreamingDataset[ChatTemplate]
