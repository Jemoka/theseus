"""
Hardware allocation solver.

Given a HardwareRequest and DispatchConfig, find the best allocation.
"""

from dataclasses import dataclass

from loguru import logger

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
    1. Check all hosts in priority order (plain and SLURM equally)
    2. Return first host that can satisfy the request
    3. If none satisfy, fall back to SLURM with most availability

    Args:
        request: Hardware requirements (chip type, min chips)
        config: Dispatch configuration with hosts
        check_availability: If True, query SLURM for real-time availability
        timeout: SSH timeout for availability checks

    Returns:
        SolveResult with allocation details, or None if unsatisfiable
    """
    logger.info(f"SOLVE | solving for {request.min_chips}x {request.chip.name}")
    inventory = RemoteInventory(config)
    chip_name = request.chip.name

    # Build host ordering: priority list first, then remaining hosts
    ordered_hosts = list(config.priority)
    for host in config.hosts:
        if host not in ordered_hosts:
            ordered_hosts.append(host)

    logger.debug(f"SOLVE | checking hosts in order: {ordered_hosts}")

    # Track SLURM candidates for fallback (with their availability)
    slurm_fallback: list[tuple[str, SlurmHostConfig, int, str | None]] = []

    # Check all hosts in priority order
    for host_name in ordered_hosts:
        if host_name not in config.hosts:
            logger.debug(f"SOLVE | skipping unknown host '{host_name}'")
            continue

        host_cfg = config.hosts[host_name]

        if isinstance(host_cfg, PlainHostConfig):
            available = host_cfg.chips.get(chip_name, 0)
            logger.debug(
                f"SOLVE | plain host '{host_name}': {available} {chip_name} available"
            )
            if available >= request.min_chips:
                logger.info(
                    f"SOLVE | selected plain host '{host_name}' with {available} chips"
                )
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

        elif isinstance(host_cfg, SlurmHostConfig):
            # Get partition (prefer default, or first)
            partition = None
            for p in host_cfg.partitions:
                if p.default:
                    partition = p.name
                    break
            if partition is None and host_cfg.partitions:
                partition = host_cfg.partitions[0].name

            if check_availability:
                logger.debug(
                    f"SOLVE | checking SLURM availability on '{host_name}' partition='{partition}'"
                )
                available = _check_slurm_availability(
                    host_cfg.ssh, partition, chip_name, config.gres_mapping, timeout
                )
            else:
                # Assume SLURM can satisfy (let scheduler handle it)
                available = request.min_chips

            logger.debug(
                f"SOLVE | SLURM host '{host_name}': {available} chips available (partition={partition})"
            )

            # Track for fallback regardless of availability
            slurm_fallback.append((host_name, host_cfg, available, partition))

            if available >= request.min_chips:
                logger.info(
                    f"SOLVE | selected SLURM host '{host_name}' partition='{partition}'"
                )
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

    # Fallback: pick SLURM with most availability (even if < min_chips)
    if slurm_fallback:
        slurm_fallback.sort(key=lambda x: x[2], reverse=True)
        host_name, host_cfg, available, partition = slurm_fallback[0]

        logger.warning(
            f"SOLVE | no host satisfies request, falling back to SLURM '{host_name}' with {available} chips"
        )
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
            is_slurm=True,
            partition=partition,
        )

    # No solution found
    logger.error(f"SOLVE | no hosts available for {request.min_chips}x {chip_name}")
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
        logger.debug("SOLVE | no partition specified, returning 0 availability")
        return 0

    try:
        # Only use user-specified gres mapping
        gpu_type = gres_mapping.get(chip_name)
        logger.debug(
            f"SOLVE | querying SLURM availability: partition={partition}, gpu_type={gpu_type}"
        )
        available = available_gpus(
            partition, ssh_alias, gpu_type=gpu_type, timeout=timeout
        )
        total = sum(count for _, count in available)
        logger.debug(f"SOLVE | SLURM availability: {total} GPUs")
        return total
    except Exception as e:
        # On error, return 0 (unavailable)
        logger.warning(f"SOLVE | failed to check SLURM availability: {e}")
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
        logger.error(
            f"SOLVE | cannot satisfy request: {request.min_chips}x {request.chip.name}"
        )
        raise RuntimeError(
            f"Cannot satisfy hardware request: {request.min_chips}x {request.chip.name}. "
            f"No hosts available in config."
        )
    return result
