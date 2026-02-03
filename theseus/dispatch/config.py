"""
Remote dispatch configuration schema.

See examples/dispatch.yaml for a complete example.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from loguru import logger
from omegaconf import OmegaConf, DictConfig

from theseus.base.chip import Chip, SUPPORTED_CHIPS
from theseus.base.hardware import Cluster


@dataclass
class JuiceFSMount:
    """JuiceFS mount configuration."""

    redis_url: str
    mount_point: str
    cache_size: str | None = None
    cache_dir: str | None = None


@dataclass
class ClusterConfig:
    """Configuration for a compute cluster's paths."""

    root: str  # root directory for checkpoints, data, etc.
    work: str  # work/scratch directory
    log: str | None = None  # log directory (defaults to {work}/logs)
    mount: str | None = None  # Redis connection string for JuiceFS mount at root
    cache_size: str | None = None  # JuiceFS --cache-size (e.g., "100G")
    cache_dir: str | None = None  # JuiceFS --cache-dir path


@dataclass
class PartitionConfig:
    """Configuration for a SLURM partition."""

    name: str
    default: bool = False
    constraint: str | None = None  # SLURM --constraint


@dataclass
class PlainHostConfig:
    """Configuration for a plain SSH host (no scheduler)."""

    ssh: str  # SSH config alias
    cluster: str  # cluster name reference
    type: Literal["plain"] = "plain"
    chips: dict[str, int] = field(default_factory=dict)  # chip_name -> count
    uv_groups: list[str] = field(default_factory=list)  # uv sync --group flags


@dataclass
class SlurmHostConfig:
    """Configuration for a SLURM cluster login node."""

    ssh: str  # SSH config alias
    cluster: str  # cluster name reference
    type: Literal["slurm"] = "slurm"
    partitions: list[PartitionConfig] = field(default_factory=list)
    account: str | None = None  # default SLURM account
    qos: str | None = None  # default QoS
    uv_groups: list[str] = field(default_factory=list)  # uv sync --group flags


@dataclass
class DispatchConfig:
    """Top-level dispatch configuration."""

    clusters: dict[str, ClusterConfig] = field(default_factory=dict)
    hosts: dict[str, PlainHostConfig | SlurmHostConfig] = field(default_factory=dict)
    priority: list[str] = field(default_factory=list)  # host names in priority order
    gres_mapping: dict[str, str] = field(
        default_factory=dict
    )  # chip name -> SLURM gres type


def load_dispatch_config(path: str | Path) -> DispatchConfig:
    """Load dispatch configuration from a YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        Parsed DispatchConfig
    """
    logger.info(f"CONFIG | loading dispatch config from {path}")
    cfg = OmegaConf.load(path)
    return parse_dispatch_config(cfg)


def parse_dispatch_config(cfg: DictConfig) -> DispatchConfig:
    """Parse dispatch configuration from OmegaConf.

    Args:
        cfg: OmegaConf config with 'clusters' and 'hosts' keys

    Returns:
        Parsed DispatchConfig
    """
    logger.debug("CONFIG | parsing dispatch config")
    clusters: dict[str, ClusterConfig] = {}
    for name, cluster_cfg in cfg.get("clusters", {}).items():
        clusters[name] = ClusterConfig(
            root=cluster_cfg.root,
            work=cluster_cfg.work,
            log=cluster_cfg.get("log"),
            mount=cluster_cfg.get("mount"),
            cache_size=cluster_cfg.get("cache_size"),
            cache_dir=cluster_cfg.get("cache_dir"),
        )
    logger.debug(f"CONFIG | parsed {len(clusters)} clusters")

    hosts: dict[str, PlainHostConfig | SlurmHostConfig] = {}
    for name, host_cfg in cfg.get("hosts", {}).items():
        host_type = host_cfg.get("type", "plain")

        uv_groups = list(host_cfg.get("uv_groups", []))

        if host_type == "plain":
            chips = dict(host_cfg.get("chips", {}))
            hosts[name] = PlainHostConfig(
                ssh=host_cfg.ssh,
                cluster=host_cfg.cluster,
                type="plain",
                chips=chips,
                uv_groups=uv_groups,
            )
        elif host_type == "slurm":
            partitions = []
            for p in host_cfg.get("partitions", []):
                if isinstance(p, str):
                    partitions.append(PartitionConfig(name=p))
                else:
                    partitions.append(
                        PartitionConfig(
                            name=p.name,
                            default=p.get("default", False),
                            constraint=p.get("constraint"),
                        )
                    )
            hosts[name] = SlurmHostConfig(
                ssh=host_cfg.ssh,
                cluster=host_cfg.cluster,
                type="slurm",
                partitions=partitions,
                account=host_cfg.get("account"),
                qos=host_cfg.get("qos"),
                uv_groups=uv_groups,
            )

    plain_count = sum(1 for h in hosts.values() if isinstance(h, PlainHostConfig))
    slurm_count = sum(1 for h in hosts.values() if isinstance(h, SlurmHostConfig))
    logger.debug(
        f"CONFIG | parsed {len(hosts)} hosts ({plain_count} plain, {slurm_count} slurm)"
    )

    priority = list(cfg.get("priority", []))
    gres_mapping = dict(cfg.get("gres_mapping", {}))

    logger.info(
        f"CONFIG | loaded config with {len(clusters)} clusters, {len(hosts)} hosts"
    )
    return DispatchConfig(
        clusters=clusters, hosts=hosts, priority=priority, gres_mapping=gres_mapping
    )


class RemoteInventory:
    """Resolves dispatch config into usable objects."""

    def __init__(self, config: DispatchConfig):
        self.config = config
        self._clusters: dict[str, Cluster] = {}

    def get_cluster(self, name: str) -> Cluster:
        """Get or create a Cluster object by name."""
        if name not in self._clusters:
            if name not in self.config.clusters:
                raise KeyError(f"Unknown cluster: {name}")
            cfg = self.config.clusters[name]
            self._clusters[name] = Cluster(
                name=name,
                root=cfg.root,
                work=cfg.work,
                log=cfg.log,
            )
        return self._clusters[name]

    def get_host(self, name: str) -> PlainHostConfig | SlurmHostConfig:
        """Get host configuration by name."""
        if name not in self.config.hosts:
            raise KeyError(f"Unknown host: {name}")
        return self.config.hosts[name]

    def get_chip(self, name: str) -> Chip:
        """Get a Chip by name from SUPPORTED_CHIPS."""
        if name not in SUPPORTED_CHIPS:
            raise KeyError(
                f"Unknown chip: {name}. Available: {list(SUPPORTED_CHIPS.keys())}"
            )
        return SUPPORTED_CHIPS[name]

    def plain_hosts(self) -> list[str]:
        """List all plain SSH hosts."""
        return [name for name, cfg in self.config.hosts.items() if cfg.type == "plain"]

    def slurm_hosts(self) -> list[str]:
        """List all SLURM hosts."""
        return [name for name, cfg in self.config.hosts.items() if cfg.type == "slurm"]

    def hosts_with_chip(self, chip: str | Chip) -> list[str]:
        """Find plain hosts that have a specific chip type."""
        chip_name = chip if isinstance(chip, str) else chip.name
        results = []
        for name, cfg in self.config.hosts.items():
            if isinstance(cfg, PlainHostConfig):
                if chip_name in cfg.chips and cfg.chips[chip_name] > 0:
                    results.append(name)
        return results

    def total_chips(self, chip: str | Chip) -> int:
        """Get total count of a chip type across all plain hosts."""
        chip_name = chip if isinstance(chip, str) else chip.name
        total = 0
        for cfg in self.config.hosts.values():
            if isinstance(cfg, PlainHostConfig):
                total += cfg.chips.get(chip_name, 0)
        return total

    def default_partition(self, host: str) -> PartitionConfig | None:
        """Get the default partition for a SLURM host."""
        cfg = self.get_host(host)
        if not isinstance(cfg, SlurmHostConfig):
            return None
        for p in cfg.partitions:
            if p.default:
                return p
        return cfg.partitions[0] if cfg.partitions else None


def discover_plain_host(ssh_alias: str, timeout: float = 30.0) -> dict[str, int]:
    """Discover chips on a plain SSH host by querying nvidia-smi.

    Args:
        ssh_alias: SSH config alias
        timeout: SSH timeout

    Returns:
        Dict mapping chip name -> count
    """
    from theseus.dispatch.ssh import run

    logger.info(f"CONFIG | discovering GPUs on {ssh_alias}")
    result = run(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits",
        ssh_alias,
        timeout=timeout,
    )

    if not result.ok:
        logger.warning(f"CONFIG | nvidia-smi failed on {ssh_alias}: {result.stderr}")
        return {}

    chips: dict[str, int] = {}
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        gpu_name, mem_mb = parts[0], int(parts[1])

        # Try to match to known chip
        matched = _match_gpu_to_chip(gpu_name, mem_mb)
        if matched:
            chips[matched] = chips.get(matched, 0) + 1
        else:
            logger.debug(f"CONFIG | unmatched GPU: {gpu_name} ({mem_mb}MB)")

    logger.info(f"CONFIG | discovered on {ssh_alias}: {chips}")
    return chips


def _match_gpu_to_chip(gpu_name: str, mem_mb: int) -> str | None:
    """Match nvidia-smi GPU name to a SUPPORTED_CHIPS key."""
    gpu_lower = gpu_name.lower()

    # Direct matches
    if "h200" in gpu_lower:
        return "h200"
    if "h100" in gpu_lower:
        return "h100"
    if "a100" in gpu_lower:
        if mem_mb >= 79000:
            return "a100-sxm4-80gb"
        else:
            return "a100-pcie-40gb"
    if "a6000" in gpu_lower:
        return "a6000"
    if "l40s" in gpu_lower:
        return "l40s"
    if "l40" in gpu_lower:
        return "l40"

    return None


def discover_slurm_partitions(
    ssh_alias: str, timeout: float = 30.0
) -> list[PartitionConfig]:
    """Discover SLURM partitions on a host.

    Args:
        ssh_alias: SSH config alias
        timeout: SSH timeout

    Returns:
        List of discovered PartitionConfig
    """
    from theseus.dispatch.slurm import partitions

    logger.info(f"CONFIG | discovering SLURM partitions on {ssh_alias}")
    parts = partitions(ssh_alias, timeout=timeout)

    configs = []
    for p in parts:
        configs.append(
            PartitionConfig(
                name=p.name,
                default=False,  # can't detect default from sinfo
            )
        )

    logger.info(f"CONFIG | discovered {len(configs)} partitions on {ssh_alias}")
    return configs
