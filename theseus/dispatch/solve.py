"""
Hardware allocation solver.

Given a HardwareRequest and DispatchConfig, find the best allocation.
"""

from dataclasses import dataclass

from theseus.base.hardware import ClusterMachine, HardwareRequest, HardwareResult
from theseus.dispatch.config import (
    DispatchConfig,
    PlainHostConfig,
    SlurmHostConfig,
    RemoteInventory,
)


@dataclass
class SolveResult:
    """Result of solving for hardware allocation."""

    result: HardwareResult | None
    host_name: str | None  # name of selected host in config
    host_config: PlainHostConfig | SlurmHostConfig | None
    is_slurm: bool  # whether allocation requires SLURM submission
    partition: str | None  # SLURM partition if applicable


def solve(
    request: HardwareRequest,
    config: DispatchConfig,
    check_availability: bool = False,
    timeout: float = 30.0,
) -> SolveResult:
    """Solve for hardware allocation given a request and config.

    Strategy:
    1. Check plain hosts (in priority order) for matching chips
    2. Fall back to SLURM clusters (in priority order)
    3. For SLURM, optionally check real-time GPU availability

    Args:
        request: Hardware requirements (chip type, min chips)
        config: Dispatch configuration with hosts
        check_availability: If True, query SLURM for real-time availability
        timeout: SSH timeout for availability checks

    Returns:
        SolveResult with allocation details, or None if unsatisfiable
    """
    inventory = RemoteInventory(config)
    chip_name = request.chip.name

    # Build host ordering: priority list first, then remaining hosts
    ordered_hosts = list(config.priority)
    for host in config.hosts:
        if host not in ordered_hosts:
            ordered_hosts.append(host)

    # Phase 1: Check plain hosts with static chip counts
    for host_name in ordered_hosts:
        if host_name not in config.hosts:
            continue

        host_cfg = config.hosts[host_name]
        if not isinstance(host_cfg, PlainHostConfig):
            continue

        available = host_cfg.chips.get(chip_name, 0)
        if available >= request.min_chips:
            cluster = inventory.get_cluster(host_cfg.cluster)
            machine = ClusterMachine(
                name=host_name,
                cluster=cluster,
                resources={request.chip: available},
            )
            return SolveResult(
                result=HardwareResult(
                    chip=request.chip,
                    hosts=[machine],
                    total_chips=available,
                ),
                host_name=host_name,
                host_config=host_cfg,
                is_slurm=False,
                partition=None,
            )

    # Phase 2: Check SLURM clusters
    slurm_candidates: list[tuple[str, SlurmHostConfig, int, str | None]] = []

    for host_name in ordered_hosts:
        if host_name not in config.hosts:
            continue

        host_cfg = config.hosts[host_name]
        if not isinstance(host_cfg, SlurmHostConfig):
            continue

        # Get partition (prefer default, or first)
        partition = None
        for p in host_cfg.partitions:
            if p.default:
                partition = p.name
                break
        if partition is None and host_cfg.partitions:
            partition = host_cfg.partitions[0].name

        if check_availability:
            # Query real-time availability
            available = _check_slurm_availability(
                host_cfg.ssh, partition, chip_name, config.gres_mapping, timeout
            )
        else:
            # Assume SLURM can satisfy (let scheduler handle it)
            available = request.min_chips

        if available >= request.min_chips:
            slurm_candidates.append((host_name, host_cfg, available, partition))

    # Pick best SLURM candidate (most availability)
    if slurm_candidates:
        slurm_candidates.sort(key=lambda x: x[2], reverse=True)
        host_name, host_cfg, available, partition = slurm_candidates[0]

        cluster = inventory.get_cluster(host_cfg.cluster)
        machine = ClusterMachine(
            name=host_name,
            cluster=cluster,
            resources={request.chip: available},
        )
        return SolveResult(
            result=HardwareResult(
                chip=request.chip,
                hosts=[machine],
                total_chips=min(available, request.min_chips),
            ),
            host_name=host_name,
            host_config=host_cfg,
            is_slurm=True,
            partition=partition,
        )

    # No solution found
    return SolveResult(
        result=None,
        host_name=None,
        host_config=None,
        is_slurm=False,
        partition=None,
    )


def _check_slurm_availability(
    ssh_alias: str,
    partition: str | None,
    chip_name: str,
    gres_mapping: dict[str, str],
    timeout: float,
) -> int:
    """Check real-time GPU availability on a SLURM cluster.

    Returns total available GPUs of the requested type.
    """
    from theseus.dispatch.slurm import available_gpus

    if partition is None:
        return 0

    try:
        # Only use user-specified gres mapping
        gpu_type = gres_mapping.get(chip_name)
        available = available_gpus(
            partition, ssh_alias, gpu_type=gpu_type, timeout=timeout
        )
        return sum(count for _, count in available)
    except Exception:
        # On error, return 0 (unavailable)
        return 0


def solve_or_raise(
    request: HardwareRequest,
    config: DispatchConfig,
    check_availability: bool = False,
    timeout: float = 30.0,
) -> SolveResult:
    """Like solve(), but raises if no solution found."""
    result = solve(request, config, check_availability, timeout)
    if result.result is None:
        raise RuntimeError(
            f"Cannot satisfy hardware request: {request.min_chips}x {request.chip.name}. "
            f"No hosts available in config."
        )
    return result
