"""
Remote dispatch utilities for SSH and SLURM.
"""

from theseus.dispatch.ssh import (
    RunResult,
    TunnelResult,
    hosts,
    run,
    run_many,
    copy_to,
    copy_from,
    is_reachable,
    forward_port,
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
    first_node_from_nodelist,
    wait_until_running,
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

from theseus.dispatch.dispatch import dispatch, dispatch_repl, ReplResult

__all__ = [
    # SSH
    "RunResult",
    "TunnelResult",
    "hosts",
    "run",
    "run_many",
    "copy_to",
    "copy_from",
    "is_reachable",
    "forward_port",
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
    "first_node_from_nodelist",
    "wait_until_running",
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
    "dispatch_repl",
    "ReplResult",
]
