"""
Remote dispatch utilities for SSH and SLURM.
"""

from theseus.dispatch.ssh import (
    RunResult,
    hosts,
    run,
    run_many,
    copy_to,
    copy_from,
    is_reachable,
)

from theseus.dispatch.slurm import (
    SlurmJob,
    SlurmResult,
    JobStatus,
    JobInfo,
    QueueResult,
    StatusResult,
    JobInfoResult,
    NodeGres,
    NodeInfo,
    PartitionInfo,
    submit,
    submit_packed,
    status,
    cancel,
    job_info,
    queue,
    wait,
    partitions,
    partition_nodes,
    node_info,
    nodes_info,
    available_gpus,
)

from theseus.dispatch.config import (
    ClusterConfig,
    PartitionConfig,
    PlainHostConfig,
    SlurmHostConfig,
    DispatchConfig,
    RemoteInventory,
    load_dispatch_config,
    parse_dispatch_config,
    discover_plain_host,
    discover_slurm_partitions,
)

from theseus.dispatch.sync import (
    snapshot,
    ship,
    ship_dirty,
    ship_files,
    sync,
)

from theseus.dispatch.solve import (
    SolveResult,
    solve,
    solve_or_raise,
)

from theseus.dispatch.dispatch import dispatch

__all__ = [
    # SSH
    "RunResult",
    "hosts",
    "run",
    "run_many",
    "copy_to",
    "copy_from",
    "is_reachable",
    # SLURM
    "SlurmJob",
    "SlurmResult",
    "JobStatus",
    "JobInfo",
    "QueueResult",
    "StatusResult",
    "JobInfoResult",
    "NodeGres",
    "NodeInfo",
    "PartitionInfo",
    "submit",
    "submit_packed",
    "status",
    "cancel",
    "job_info",
    "queue",
    "wait",
    "partitions",
    "partition_nodes",
    "node_info",
    "nodes_info",
    "available_gpus",
    # Config
    "ClusterConfig",
    "PartitionConfig",
    "PlainHostConfig",
    "SlurmHostConfig",
    "DispatchConfig",
    "RemoteInventory",
    "load_dispatch_config",
    "parse_dispatch_config",
    "discover_plain_host",
    "discover_slurm_partitions",
    # Sync
    "snapshot",
    "ship",
    "ship_dirty",
    "ship_files",
    "sync",
    # Solve
    "SolveResult",
    "solve",
    "solve_or_raise",
    # Dispatch
    "dispatch",
]
