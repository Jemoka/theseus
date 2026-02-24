"""
Hardware allocation solver.

Given a HardwareRequest and DispatchConfig, find the best allocation.
"""

from dataclasses import dataclass

from loguru import logger

from theseus.base.hardware import ClusterMachine, HardwareRequest, HardwareResult
from theseus.dispatch.config import (
    DispatchConfig,
    PartitionConfig,
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
        request: Hardware requirements (chip type, min chips, cluster preferences)
        config: Dispatch configuration with hosts
        check_availability: If True, query SLURM for real-time availability
        timeout: SSH timeout for availability checks

    Returns:
        SolveResult with allocation details, or None if unsatisfiable
    """
    chip_name = request.chip.name if request.chip else None
    cpu_mode = request.min_chips == 0
    target_desc = (
        "cpu-only" if cpu_mode else f"{request.min_chips}x {chip_name or 'any-gpu'}"
    )
    logger.info(f"SOLVE | solving for {target_desc}")
    inventory = RemoteInventory(config)

    # Build host ordering: priority list first, then remaining hosts
    ordered_hosts = list(config.priority)
    for host in config.hosts:
        if host not in ordered_hosts:
            ordered_hosts.append(host)

    # Filter hosts by cluster membership (from HardwareRequest)
    preferred_set = (
        set(request.preferred_clusters) if request.preferred_clusters else None
    )
    forbidden_set = (
        set(request.forbidden_clusters) if request.forbidden_clusters else set()
    )

    if preferred_set or forbidden_set:
        filtered_hosts = []
        for host_name in ordered_hosts:
            if host_name not in config.hosts:
                continue
            host_cfg = config.hosts[host_name]
            host_cluster = host_cfg.cluster

            if host_cluster in forbidden_set:
                logger.debug(
                    f"SOLVE | excluding host '{host_name}' (cluster '{host_cluster}' is forbidden)"
                )
                continue
            if preferred_set and host_cluster not in preferred_set:
                logger.debug(
                    f"SOLVE | skipping host '{host_name}' (cluster '{host_cluster}' not in preferred list)"
                )
                continue
            filtered_hosts.append(host_name)

        ordered_hosts = filtered_hosts
        logger.debug(f"SOLVE | after cluster filtering: {ordered_hosts}")

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
            if cpu_mode:
                logger.info(
                    f"SOLVE | selected plain host '{host_name}' for cpu-only request"
                )
                cluster = inventory.get_cluster(host_cfg.cluster)
                machine = ClusterMachine(name=host_name, cluster=cluster, resources={})
                return SolveResult(
                    result=HardwareResult(chip=None, hosts=[machine], total_chips=0),
                    host_name=host_name,
                    host_config=host_cfg,
                    is_slurm=False,
                    partition=None,
                )

            if chip_name is None:
                configured = sum(max(v, 0) for v in host_cfg.chips.values())
                configured_label = "any-gpu"
            else:
                configured = host_cfg.chips.get(chip_name, 0)
                configured_label = chip_name

            if configured == 0:
                logger.debug(
                    f"SOLVE | plain host '{host_name}': no {configured_label} configured"
                )
                continue

            # Check real-time availability if requested
            if check_availability:
                logger.debug(f"SOLVE | checking GPU availability on '{host_name}'")
                available = _check_plain_host_availability(
                    host_cfg.ssh, configured, timeout
                )
            else:
                available = configured

            logger.debug(
                f"SOLVE | plain host '{host_name}': {available}/{configured} {configured_label} available"
            )
            if available >= request.min_chips:
                logger.info(
                    f"SOLVE | selected plain host '{host_name}' with {available} chips"
                )
                cluster = inventory.get_cluster(host_cfg.cluster)
                machine_resources = (
                    {request.chip: available} if request.chip is not None else {}
                )
                machine = ClusterMachine(
                    name=host_name,
                    cluster=cluster,
                    resources=machine_resources,
                )
                return SolveResult(
                    result=HardwareResult(
                        chip=request.chip,
                        hosts=[machine],
                        total_chips=request.min_chips,
                    ),
                    host_name=host_name,
                    host_config=host_cfg,
                    is_slurm=False,
                    partition=None,
                )

        elif isinstance(host_cfg, SlurmHostConfig):
            # Check if this host has an explicit chip limit
            chip_limit = None
            if not cpu_mode and host_cfg.chips is not None:
                if chip_name is None:
                    chip_limit = sum(max(v, 0) for v in host_cfg.chips.values())
                    if chip_limit == 0:
                        logger.debug(
                            f"SOLVE | SLURM host '{host_name}': no allowed GPU chips"
                        )
                        continue
                    logger.debug(
                        f"SOLVE | SLURM host '{host_name}': total chip limit = {chip_limit}"
                    )
                else:
                    chip_limit = host_cfg.chips.get(chip_name, 0)
                    if chip_limit == 0:
                        logger.debug(
                            f"SOLVE | SLURM host '{host_name}': chip '{chip_name}' not in allowed chips list"
                        )
                        continue
                    logger.debug(
                        f"SOLVE | SLURM host '{host_name}': chip limit for {chip_name} = {chip_limit}"
                    )

            # Check all partitions and pick the one with most availability
            best_partition = None
            best_available = 0

            eligible_partitions: list[PartitionConfig] = []
            cpu_partition_names: list[str] = []
            if cpu_mode:
                # CPU mode uses explicit cpu_partitions when provided.
                # If omitted/empty, fall back to regular partitions order.
                candidate_names = (
                    list(host_cfg.cpu_partitions)
                    if host_cfg.cpu_partitions
                    else [p.name for p in host_cfg.partitions]
                )
                # De-duplicate while preserving order
                seen: set[str] = set()
                cpu_partition_names = []
                for name in candidate_names:
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    cpu_partition_names.append(name)
                if not cpu_partition_names:
                    logger.debug(
                        f"SOLVE | SLURM host '{host_name}': no cpu_partitions/partitions configured"
                    )
                    continue
            else:
                # Get the GRES type name for this chip (None => any GPU type)
                gres_type = config.gres_mapping.get(chip_name) if chip_name else None

                # Query GPU types for all partitions on this host (single SSH call)
                from theseus.dispatch.slurm import partition_gpu_types

                partition_names_to_check = [p.name for p in host_cfg.partitions]
                gpu_types_by_partition = partition_gpu_types(
                    host_cfg.ssh, partition_names_to_check, timeout=timeout
                )

                # Build partition list: filter by requested/available GPU type, default first
                for p in host_cfg.partitions:
                    partition_types = gpu_types_by_partition.get(p.name, set())
                    if gres_type and gres_type not in partition_types:
                        logger.debug(
                            f"SOLVE | skipping partition '{p.name}': doesn't have {gres_type} (has: {partition_types})"
                        )
                        continue
                    if chip_name is None and not partition_types:
                        logger.debug(
                            f"SOLVE | skipping partition '{p.name}': no GPUs detected for generic GPU request"
                        )
                        continue
                    if p.default:
                        eligible_partitions.insert(0, p)
                    else:
                        eligible_partitions.append(p)

                if not eligible_partitions:
                    target = gres_type or chip_name or "any-gpu"
                    logger.debug(
                        f"SOLVE | SLURM host '{host_name}': no partitions have {target}"
                    )
                    continue

            partition_names = (
                cpu_partition_names
                if cpu_mode
                else [p.name for p in eligible_partitions]
            )

            if cpu_mode:
                available = 0
                partition = partition_names[0] if partition_names else None
            elif check_availability:
                for part_name in partition_names:
                    logger.debug(
                        f"SOLVE | checking SLURM availability on '{host_name}' partition='{part_name}'"
                    )
                    avail = _check_slurm_availability(
                        host_cfg.ssh,
                        part_name,
                        chip_name,
                        config.gres_mapping,
                        timeout,
                    )
                    # Apply chip limit if specified
                    if chip_limit is not None:
                        avail = min(avail, chip_limit)
                    logger.debug(
                        f"SOLVE | partition '{part_name}': {avail} chips available"
                    )
                    if avail > best_available:
                        best_available = avail
                        best_partition = part_name
                    # Early exit if we found enough
                    if avail >= request.min_chips:
                        break

                available = best_available
                partition = best_partition or (
                    partition_names[0] if partition_names else None
                )
            else:
                # Assume SLURM can satisfy (let scheduler handle it)
                # But cap at chip limit if specified
                available = chip_limit if chip_limit is not None else request.min_chips
                partition = partition_names[0] if partition_names else None

            logger.debug(
                f"SOLVE | SLURM host '{host_name}': {available} chips available (partition={partition})"
            )

            # Track for fallback - partition is known to have this chip type
            slurm_fallback.append((host_name, host_cfg, available, partition))

            if cpu_mode or available >= request.min_chips:
                logger.info(
                    f"SOLVE | selected SLURM host '{host_name}' partition='{partition}'"
                )
                cluster = inventory.get_cluster(host_cfg.cluster)
                machine_resources = (
                    {request.chip: available}
                    if request.chip is not None and available > 0
                    else {}
                )
                machine = ClusterMachine(
                    name=host_name,
                    cluster=cluster,
                    resources=machine_resources,
                )
                return SolveResult(
                    result=HardwareResult(
                        chip=request.chip,
                        hosts=[machine],
                        total_chips=request.min_chips,
                    ),
                    host_name=host_name,
                    host_config=host_cfg,
                    is_slurm=True,
                    partition=partition,
                )

    # Fallback: pick SLURM with most availability (even if < min_chips)
    # For SLURM, we request the full min_chips and let the scheduler queue until available
    # (but respect chip limit if configured)
    if slurm_fallback:
        slurm_fallback.sort(key=lambda x: x[2], reverse=True)
        host_name, host_cfg, available, partition = slurm_fallback[0]

        # Determine request amount: cap at chip limit if specified
        fallback_chip_limit = None
        if host_cfg.chips is not None and not cpu_mode:
            if chip_name is None:
                fallback_chip_limit = sum(max(v, 0) for v in host_cfg.chips.values())
            else:
                fallback_chip_limit = host_cfg.chips.get(chip_name, 0)
        chips_to_request = request.min_chips
        if fallback_chip_limit is not None:
            chips_to_request = min(chips_to_request, fallback_chip_limit)

        logger.warning(
            f"SOLVE | no host satisfies request, falling back to SLURM '{host_name}' "
            f"(available={available}, requesting={chips_to_request})"
        )
        cluster = inventory.get_cluster(host_cfg.cluster)
        machine = ClusterMachine(
            name=host_name,
            cluster=cluster,
            resources={request.chip: chips_to_request}
            if request.chip is not None and chips_to_request > 0
            else {},
        )
        return SolveResult(
            result=HardwareResult(
                chip=request.chip,
                hosts=[machine],
                total_chips=chips_to_request,  # Request amount (may be capped by chip limit)
            ),
            host_name=host_name,
            host_config=host_cfg,
            is_slurm=True,
            partition=partition,
        )

    # No solution found
    logger.error(f"SOLVE | no hosts available for {target_desc}")
    return SolveResult(
        result=None,
        host_name=None,
        host_config=None,
        is_slurm=False,
        partition=None,
    )


def _check_plain_host_availability(
    ssh_alias: str,
    configured_chips: int,
    timeout: float,
) -> int:
    """Check if GPUs on a plain SSH host are available (no running processes).

    Runs nvidia-smi to check if any GPU has processes. If processes are running,
    returns 0 (host is busy). Otherwise returns the configured chip count.

    Args:
        ssh_alias: SSH config alias
        configured_chips: Number of chips configured for this host
        timeout: SSH timeout

    Returns:
        Available chip count (0 if busy, configured_chips if free)
    """
    from theseus.dispatch.ssh import run

    # Processes below this threshold are considered noise (e.g., Xorg display server
    # at ~4 MiB) and do not count as the GPU being occupied.
    _NOISE_THRESHOLD_MIB = 100

    try:
        # Query nvidia-smi for processes on each GPU, including memory usage
        result = run(
            "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits",
            ssh_alias,
            timeout=timeout,
        )

        if not result.ok:
            # nvidia-smi failed - maybe no CUDA, treat as unavailable
            logger.warning(f"SOLVE | nvidia-smi failed on {ssh_alias}: {result.stderr}")
            return 0

        # If output is empty, no processes running - GPUs are free
        if not result.stdout.strip():
            logger.debug(f"SOLVE | {ssh_alias}: no GPU processes running, host is free")
            return configured_chips

        # Filter out low-memory processes (display servers, etc.)
        significant = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(",")
            try:
                raw = parts[3].strip() if len(parts) >= 4 else ""
                mem_mib = int(raw) if raw else None
            except ValueError:
                mem_mib = None  # "N/A" or unparseable
            # Only significant if we got a real value at or above the threshold.
            # Zero, unparseable, or "N/A" (e.g. fused arch like GB10) → noise.
            if mem_mib is not None and mem_mib >= _NOISE_THRESHOLD_MIB:
                significant.append(line.strip())
            else:
                logger.debug(
                    f"SOLVE | {ssh_alias}: ignoring low-memory process ({mem_mib} MiB): {line.strip()}"
                )

        if significant:
            logger.debug(
                f"SOLVE | {ssh_alias}: {len(significant)} significant GPU processes, host is busy"
            )
            return 0

        logger.debug(
            f"SOLVE | {ssh_alias}: only noise-level GPU processes, host is free"
        )
        return configured_chips

    except Exception as e:
        logger.warning(f"SOLVE | failed to check GPU availability on {ssh_alias}: {e}")
        return 0


def _check_slurm_availability(
    ssh_alias: str,
    partition: str | None,
    chip_name: str | None,
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
        gpu_type = gres_mapping.get(chip_name) if chip_name else None
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
        target_desc = (
            "cpu-only"
            if request.min_chips == 0
            else f"{request.min_chips}x {(request.chip.name if request.chip else 'any-gpu')}"
        )
        logger.error(f"SOLVE | cannot satisfy request: {target_desc}")
        raise RuntimeError(
            f"Cannot satisfy hardware request: {target_desc}. "
            f"No hosts available in config."
        )
    return result
