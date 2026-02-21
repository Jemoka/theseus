from typing import Generic, TypeVar, List, Dict, Iterator, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

C = TypeVar("C")

####### chat infrastructure #######


@dataclass
class ChatTurn:
    role: str
    message: str
    metadata: Dict[str, str] = field(default_factory=dict)


ChatTemplate = List[ChatTurn]

####### dataset #######


class Dataset(ABC, Generic[C]):
    @abstractmethod
    def __getitem__(self, indx: int) -> C: ...

    @abstractmethod
    def __len__(self) -> int: ...


StringDataset = Dataset[str]
ChatTemplateDataset = Dataset[ChatTemplate]


class PretrainingDataset(StringDataset):
    """
    Pretraining dataset, identical to Dataset[str] but is tokenized differently.
    Specifically, this dataset is tokenized irrespective of item boundaries.
    """

    ...


####### contrastive datasets #######


class ContrastiveDataset(ABC, Generic[C]):
    @abstractmethod
    def __getitem__(self, indx: int) -> Tuple[C, C]: ...

    @abstractmethod
    def __len__(self) -> int: ...


ContrastiveStringDataset = ContrastiveDataset[str]
ContrastiveChatTemplateDataset = ContrastiveDataset[ChatTemplate]

####### streaming datasets #######


class StreamingDataset(ABC, Generic[C]):
    @abstractmethod
    def __iter__(self) -> Iterator[C]: ...


StreamingStringDataset = StreamingDataset[str]
StreamingChatTemplateDataset = StreamingDataset[ChatTemplate]


class StreamingPretrainingDataset(StreamingStringDataset):
    """
    Pretraining dataset, identical to Dataset[str] but is tokenized differently.
    Specifically, this dataset is tokenized irrespective of item boundaries.
    """

    ...
