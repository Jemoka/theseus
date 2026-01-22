from enum import Enum


class Axis(Enum):
    BATCH = "batch"  # anything that should have "rare comms", i.e. data parallel
    SHARD = "shard"  # anything that should have "frequent comms", i.e. tensor parallel
