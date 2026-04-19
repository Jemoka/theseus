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
    all_squash: str | None = None  # passthrough for juicefs `--all-squash UID:GID`


@dataclass
class ClusterConfig:
    """Configuration for a compute cluster's paths."""

    root: str  # root directory for checkpoints, data, etc.
    work: str  # work/scratch directory
    log: str | None = None  # log directory (defaults to {work}/logs)
    data: str | None = None  # data directory (defaults to {root}/data)
    checkpoints: str | None = (
        None  # checkpoints directory (defaults to {root}/checkpoints)
    )
    results: str | None = None  # results directory (defaults to {root}/results)
    status: str | None = None  # status directory (defaults to {root}/status)
    share: str | None = (
        None  # shared temp dir visible to all nodes (defaults to {work}/.dispatch)
    )
    mount: str | None = None  # Redis connection string for JuiceFS mount at root
    cache_size: str | None = None  # JuiceFS --cache-size (e.g., "100G")
    cache_dir: str | None = None  # JuiceFS --cache-dir path
    all_squash: str | None = None  # JuiceFS --all-squash UID:GID (e.g., "1000:1000")
    uv_dir: str | None = None  # UV_CACHE_DIR override for uv
    wandb: str | None = None  # W&B API key → exported as WANDB_API_KEY
    wandb_entity: str | None = None  # W&B entity → exported as WANDB_ENTITY
    wandb_project: str | None = None  # W&B project → exported as WANDB_PROJECT
    hf_token: str | None = None  # HuggingFace token → exported as HF_TOKEN


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
    env: dict[str, str] = field(default_factory=dict)  # host-level env vars


@dataclass
class SlurmHostConfig:
    """Configuration for a SLURM cluster login node."""

    ssh: str  # SSH config alias
    cluster: str  # cluster name reference
    type: Literal["slurm"] = "slurm"
    partitions: list[PartitionConfig] = field(default_factory=list)
    account: str | None = None  # default SLURM account
    qos: str | None = None  # default QoS
    mem: str | None = None  # default memory (e.g., "64G"), defaults to 64G if not set
    exclude: list[str] = field(default_factory=list)  # nodes to exclude (--exclude)
    uv_groups: list[str] = field(default_factory=list)  # uv sync --group flags
    chips: dict[str, int] | None = (
        None  # optional chip_name -> count (limits allocation)
    )
    cpu_partitions: list[str] = field(
        default_factory=list
    )  # optional CPU-only partition preference order
    annotations: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)  # host-level env vars


@dataclass
class TPUHostConfig:
    """Configuration for a Google Cloud TPU VM.

    The host name (dict key in ``hosts:``) is used as the TPU VM name in
    ``gcloud`` commands.
    """

    cluster: str  # cluster name reference
    type: Literal["tpu"] = "tpu"
    zone: str = ""  # GCP zone (e.g., "us-central2-b")
    project: str | None = None  # GCP project (defaults to gcloud default)
    accelerator_type: str = ""  # e.g., "v4-32", "v5e-16"
    version: str = ""  # TPU software/runtime version
    spot: bool = False  # use Spot VM pricing
    preemptible: bool = False  # use preemptible pricing
    network: str | None = None  # VPC network
    subnetwork: str | None = None  # VPC subnetwork
    service_account: str | None = None  # GCP service account
    internal_ip: bool = False  # use internal IP for SSH/SCP
    metadata: dict[str, str] = field(default_factory=dict)  # instance metadata
    uv_groups: list[str] = field(default_factory=list)  # uv sync --group flags
    env: dict[str, str] = field(default_factory=dict)  # host-level env vars


@dataclass
class VolcanoHostConfig:
    """Configuration for a Kubernetes Volcano batch scheduler host."""

    cluster: str  # cluster name reference
    type: Literal["volcano"] = "volcano"
    namespace: str = "default"
    queue: str = ""
    image: str = ""  # container image
    pvc_name: str = ""  # PVC for code + data
    pvc_mount_path: str = "/workspace"  # mount point in pods
    service_account: str | None = None
    node_selector: dict[str, str] = field(default_factory=dict)
    tolerations: list[dict[str, str]] = field(default_factory=list)
    chips: dict[str, int] = field(default_factory=dict)  # chip_name -> count per node
    num_nodes: int = 1
    gpus_per_node: int = 0
    gpu_resource_key: str = "nvidia.com/gpu"  # K8s resource name for GPUs
    cpu: str | None = None
    memory: str | None = None
    cpu_cpu: str | None = None  # CPU request for cpu-only jobs (n_chips=0)
    cpu_memory: str | None = None  # memory request for cpu-only jobs (n_chips=0)
    shm_size: str | None = None  # /dev/shm size (e.g. "64Gi")
    priority_class: str | None = None
    kubeconfig: str | None = None
    context: str | None = None
    rdma: bool = False  # request RDMA network devices (rdma/rdma_shared_device_a)
    rdma_per_node: int = 8  # RDMA device count per node (when rdma=True)
    labels: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    uv_groups: list[str] = field(default_factory=list)
    helper_resources: dict[str, str] = field(
        default_factory=lambda: {
            "requests.cpu": "1",
            "requests.memory": "1Gi",
            "limits.cpu": "1",
            "limits.memory": "1Gi",
        }
    )  # resource requests/limits for PVC loader helper pod


@dataclass
class DispatchConfig:
    """Top-level dispatch configuration."""

    mount: str | None = None  # Local JuiceFS mount point for mailbox sync workflows
    proxy: str | None = None  # SCP proxy root for mailbox sync workflows
    clusters: dict[str, ClusterConfig] = field(default_factory=dict)
    hosts: dict[
        str, PlainHostConfig | SlurmHostConfig | TPUHostConfig | VolcanoHostConfig
    ] = field(default_factory=dict)
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
    logger.debug("CONFIG | loading dispatch config")
    logger.debug(f"CONFIG | config path: {path}")
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
            data=cluster_cfg.get("data"),
            checkpoints=cluster_cfg.get("checkpoints"),
            results=cluster_cfg.get("results"),
            status=cluster_cfg.get("status"),
            share=cluster_cfg.get("share"),
            mount=cluster_cfg.get("mount"),
            cache_size=cluster_cfg.get("cache_size"),
            cache_dir=cluster_cfg.get("cache_dir"),
            all_squash=cluster_cfg.get("all_squash"),
            uv_dir=cluster_cfg.get("uv_dir"),
            wandb=cluster_cfg.get("wandb"),
            wandb_entity=cluster_cfg.get("wandb_entity"),
            wandb_project=cluster_cfg.get("wandb_project"),
            hf_token=cluster_cfg.get("hf_token"),
        )
    logger.debug(f"CONFIG | parsed {len(clusters)} clusters")

    hosts: dict[
        str, PlainHostConfig | SlurmHostConfig | TPUHostConfig | VolcanoHostConfig
    ] = {}
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
                env=dict(host_cfg.get("env", {})),
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
            exclude = list(host_cfg.get("exclude", []))
            chips = dict(host_cfg.get("chips", {})) if host_cfg.get("chips") else None  # type: ignore
            cpu_partitions = list(
                host_cfg.get("cpu_partitions", host_cfg.get("partitions_cpu", []))
            )
            annotations = dict(host_cfg.get("annotations", {}))
            hosts[name] = SlurmHostConfig(
                ssh=host_cfg.ssh,
                cluster=host_cfg.cluster,
                type="slurm",
                partitions=partitions,
                account=host_cfg.get("account"),
                qos=host_cfg.get("qos"),
                mem=host_cfg.get("mem"),
                exclude=exclude,
                uv_groups=uv_groups,
                chips=chips,
                cpu_partitions=cpu_partitions,
                annotations=annotations,
                env=dict(host_cfg.get("env", {})),
            )
        elif host_type == "tpu":
            metadata = dict(host_cfg.get("metadata", {}))
            hosts[name] = TPUHostConfig(
                cluster=host_cfg.cluster,
                type="tpu",
                zone=host_cfg.get("zone", ""),
                project=host_cfg.get("project"),
                accelerator_type=host_cfg.get("accelerator_type", ""),
                version=host_cfg.get("version", ""),
                spot=host_cfg.get("spot", False),
                preemptible=host_cfg.get("preemptible", False),
                network=host_cfg.get("network"),
                subnetwork=host_cfg.get("subnetwork"),
                service_account=host_cfg.get("service_account"),
                internal_ip=host_cfg.get("internal_ip", False),
                metadata=metadata,
                uv_groups=uv_groups,
                env=dict(host_cfg.get("env", {})),
            )
        elif host_type == "volcano":
            chips = dict(host_cfg.get("chips", {}))
            node_selector = dict(host_cfg.get("node_selector", {}))
            tolerations = list(host_cfg.get("tolerations", []))
            labels = dict(host_cfg.get("labels", {}))
            env = dict(host_cfg.get("env", {}))
            hosts[name] = VolcanoHostConfig(
                cluster=host_cfg.cluster,
                type="volcano",
                namespace=host_cfg.get("namespace", "default"),
                queue=host_cfg.get("queue", ""),
                image=host_cfg.get("image", ""),
                pvc_name=host_cfg.get("pvc_name", ""),
                pvc_mount_path=host_cfg.get("pvc_mount_path", "/workspace"),
                service_account=host_cfg.get("service_account"),
                node_selector=node_selector,
                tolerations=tolerations,
                chips=chips,
                num_nodes=host_cfg.get("num_nodes", 1),
                gpus_per_node=host_cfg.get("gpus_per_node", 0),
                gpu_resource_key=host_cfg.get("gpu_resource_key", "nvidia.com/gpu"),
                cpu=host_cfg.get("cpu"),
                memory=host_cfg.get("memory"),
                cpu_cpu=host_cfg.get("cpu_cpu"),
                cpu_memory=host_cfg.get("cpu_memory"),
                shm_size=host_cfg.get("shm_size"),
                priority_class=host_cfg.get("priority_class"),
                kubeconfig=host_cfg.get("kubeconfig"),
                context=host_cfg.get("context"),
                rdma=host_cfg.get("rdma", False),
                rdma_per_node=host_cfg.get("rdma_per_node", 8),
                labels=labels,
                env=env,
                uv_groups=uv_groups,
                helper_resources=dict(
                    host_cfg.get(
                        "helper_resources",
                        {
                            "requests.cpu": "1",
                            "requests.memory": "1Gi",
                            "limits.cpu": "1",
                            "limits.memory": "1Gi",
                        },
                    )
                ),
            )

    plain_count = sum(1 for h in hosts.values() if isinstance(h, PlainHostConfig))
    slurm_count = sum(1 for h in hosts.values() if isinstance(h, SlurmHostConfig))
    tpu_count = sum(1 for h in hosts.values() if isinstance(h, TPUHostConfig))
    volcano_count = sum(1 for h in hosts.values() if isinstance(h, VolcanoHostConfig))
    logger.debug(
        f"CONFIG | parsed {len(hosts)} hosts ({plain_count} plain, {slurm_count} slurm, {tpu_count} tpu, {volcano_count} volcano)"
    )

    priority = list(cfg.get("priority", []))
    gres_mapping = dict(cfg.get("gres_mapping", {}))

    logger.debug(
        f"CONFIG | loaded config with {len(clusters)} clusters, {len(hosts)} hosts"
    )
    top_mount = cfg.get("mount")
    top_proxy = cfg.get("proxy")
    if top_mount and top_proxy:
        raise ValueError(
            "dispatch config top-level keys 'mount' and 'proxy' are mutually exclusive"
        )

    return DispatchConfig(
        mount=top_mount,
        proxy=top_proxy,
        clusters=clusters,
        hosts=hosts,
        priority=priority,
        gres_mapping=gres_mapping,
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
                data=cfg.data,
                checkpoints=cfg.checkpoints,
                results=cfg.results,
                status=cfg.status,
            )
        return self._clusters[name]

    def get_host(
        self, name: str
    ) -> PlainHostConfig | SlurmHostConfig | TPUHostConfig | VolcanoHostConfig:
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
    if "drive-pg199" in gpu_lower:
        return "drive-pg199"
    if "5090" in gpu_lower:
        return "rtx5090"

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
