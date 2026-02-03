"""
Cluster information.
"""

from typing import Annotated, Any, Optional
from pydantic import BaseModel, Field, model_validator
from pathlib import Path

from theseus.base.chip import Chip


class Cluster(BaseModel):
    name: str
    root: str  # root directory of checkpoints, code, etc.
    work: str  # work directory (where mirrors will be copied)
    log: Optional[str] = None  # log directory (defaults to {work}/logs)

    @property
    def log_dir(self) -> str:
        """Log directory path, defaults to {work}/logs if not configured."""
        return self.log if self.log else f"{self.work}/logs"

    @property
    def root_dir(self) -> Path:
        # panic if not exist
        root_dir = Path(self.root)
        if not root_dir.exists():
            raise ValueError(
                f"Cluster root directory does not exist: cluster={self.name}, root={root_dir}"
            )

        return root_dir

    @property
    def data_dir(self) -> Path:
        # make directory if not exist
        data_dir = Path(self.root_dir) / "data"
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @property
    def checkpoints_dir(self) -> Path:
        # make directory if not exist
        checkpoints_dir = Path(self.root_dir) / "checkpoints"
        if not checkpoints_dir.exists():
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
        return checkpoints_dir

    @property
    def results_dir(self) -> Path:
        # make directory if not exist
        data_dir = Path(self.root_dir) / "results"
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir


class ClusterMachine(BaseModel):
    name: str
    cluster: Cluster
    resources: dict[Chip, int]


class HardwareRequest(BaseModel):
    """
    Minimal, intent-level hardware request.

    Assumptions:
      - Storage is uniform across hosts (distributed FS).
      - Interconnects / topology are cluster concerns.
    """

    chip: Annotated[
        Chip,
        Field(description="chip type"),
    ]
    min_chips: Annotated[
        int,
        Field(ge=1, description="minimum number of chips requested"),
    ]
    preferred_clusters: Annotated[
        list[str],
        Field(default_factory=list, description="clusters to prefer (by name)"),
    ]
    forbidden_clusters: Annotated[
        list[str],
        Field(default_factory=list, description="clusters to avoid (by name)"),
    ]

    @model_validator(mode="after")
    def _validate(self) -> "HardwareRequest":
        overlap = set(self.preferred_clusters) & set(self.forbidden_clusters)
        if overlap:
            raise ValueError(
                f"Clusters cannot be both preferred and forbidden: {sorted(overlap)}"
            )

        return self


class HardwareResult(BaseModel):
    """
    Concrete hardware allocation result.
    """

    chip: Annotated[
        Optional[Chip],
        Field(description="chip type"),
    ]
    hosts: Annotated[
        list[ClusterMachine],
        Field(description="allocated hosts"),
    ]
    total_chips: Annotated[
        int,
        Field(ge=0, description="total number of allocated chips"),
    ]


def _longest_common_substring(s1: str, s2: str) -> int:
    """Return length of longest common substring between s1 and s2."""
    if not s1 or not s2:
        return 0

    m, n = len(s1), len(s2)
    # Use rolling array for space efficiency
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    max_len = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
                max_len = max(max_len, curr[j])
            else:
                curr[j] = 0
        prev, curr = curr, prev

    return max_len


def _normalize_gpu_name(name: str) -> str:
    """Normalize GPU name by lowercasing and removing vendor prefixes."""
    name = name.lower()
    # Remove common vendor prefixes
    for prefix in ["nvidia", "amd", "intel", "google"]:
        name = name.replace(prefix, "")
    # Clean up whitespace
    return " ".join(name.split())


def _match_chip_from_jax_device(device: Any) -> Optional[Chip]:
    """
    Match a JAX device to a supported Chip using longest common substring.
    Returns None if no match found.
    """
    from theseus.base.chip import SUPPORTED_CHIPS

    platform = device.platform
    device_kind = device.device_kind if hasattr(device, "device_kind") else ""

    if platform == "tpu":
        device_kind_lower = device_kind.lower()
        # Match TPU by version
        for chip_key, chip in SUPPORTED_CHIPS.items():
            if chip_key.startswith("tpu-"):
                version = chip_key.replace("tpu-", "")
                if version in device_kind_lower:
                    return chip
        # Unknown TPU
        return Chip(
            name=f"tpu-{device_kind_lower.replace(' ', '-')}",
            display_name=f"Google {device_kind}",
            memory=int(16 * 1024**3),
        )

    elif platform == "gpu":
        # Normalize the device name (remove "NVIDIA" etc.)
        normalized_device = _normalize_gpu_name(device_kind)

        # Find best match using longest common substring
        best_chip: Optional[Chip] = None
        best_score = 0

        for chip_key, chip in SUPPORTED_CHIPS.items():
            # Skip non-GPU chips
            if chip_key.startswith("tpu-") or chip_key == "cpu":
                continue

            normalized_chip = _normalize_gpu_name(chip.display_name)
            score = _longest_common_substring(normalized_device, normalized_chip)

            # Require minimum overlap to consider a match
            if score >= 2 and score > best_score:
                best_score = score
                best_chip = chip

        return best_chip

    elif platform == "cpu":
        return SUPPORTED_CHIPS["cpu"]

    return None


def _get_gpu_info_from_nvidia_smi() -> Optional[Chip]:
    """
    Query nvidia-smi for detailed GPU info when JAX doesn't provide enough.
    Returns matched Chip or creates one for unknown GPU.
    """
    import subprocess

    from theseus.base.chip import SUPPORTED_CHIPS

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    if not lines:
        return None

    # Parse first GPU info
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) < 3:
        return None

    _, gpu_name, mem_mb = parts[0], parts[1], int(parts[2])

    # Match to supported chip using longest common substring
    normalized_device = _normalize_gpu_name(gpu_name)
    best_chip: Optional[Chip] = None
    best_score = 0

    for chip_key, chip in SUPPORTED_CHIPS.items():
        # Skip non-GPU chips
        if chip_key.startswith("tpu-") or chip_key == "cpu":
            continue

        normalized_chip = _normalize_gpu_name(chip.display_name)
        score = _longest_common_substring(normalized_device, normalized_chip)

        if score >= 2 and score > best_score:
            best_score = score
            best_chip = chip

    if best_chip is not None:
        return best_chip

    # Create chip for unknown GPU
    return Chip(
        name=_normalize_gpu_name(gpu_name).replace(" ", "-"),
        display_name=gpu_name,
        memory=mem_mb * 1024 * 1024,
    )


def local(root_dir: str, work_dir: str) -> HardwareResult:
    """
    Detect hardware using JAX device information.
    Creates one ClusterMachine per host, with host index matching jax.process_index().
    Assumes paths are identical across all hosts.
    """
    import jax
    from collections import defaultdict

    from theseus.base.chip import SUPPORTED_CHIPS

    devices = jax.devices()
    process_count = jax.process_count()

    # Group devices by process index to count per host
    devices_per_host: dict[int, int] = defaultdict(int)
    chip: Chip

    if not devices:
        # No devices at all, fall back to CPU
        chip = SUPPORTED_CHIPS["cpu"]
        total_chips = 1
        devices_per_host[0] = 1
    else:
        for d in devices:
            devices_per_host[d.process_index] += 1

        # Determine chip type from first device
        first_device = devices[0]
        matched_chip = _match_chip_from_jax_device(first_device)

        # If GPU and no match, try nvidia-smi for more info
        if matched_chip is None and first_device.platform == "gpu":
            matched_chip = _get_gpu_info_from_nvidia_smi()

        # Final fallback to CPU if still no chip
        if matched_chip is None:
            chip = SUPPORTED_CHIPS["cpu"]
        else:
            chip = matched_chip

        total_chips = len(devices)

    import socket

    # Create cluster (paths assumed identical across hosts)
    cluster = Cluster(name="local", root=root_dir, work=work_dir)

    # Get current host's actual hostname
    current_process_idx = jax.process_index()
    current_hostname = socket.gethostname()

    # Create one ClusterMachine per host
    # Host index matches jax.process_index()
    hosts: list[ClusterMachine] = []
    for host_idx in range(process_count):
        host_device_count = devices_per_host.get(host_idx, 0)
        resources: dict[Chip, int] = {}
        if host_device_count > 0:
            resources[chip] = host_device_count

        # Use actual hostname for current host, fallback to index for others
        if host_idx == current_process_idx:
            host_name = current_hostname
        else:
            host_name = f"host-{host_idx}"

        hosts.append(
            ClusterMachine(
                name=host_name,
                cluster=cluster,
                resources=resources,
            )
        )

    return HardwareResult(chip=chip, hosts=hosts, total_chips=total_chips)
