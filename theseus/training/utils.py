"""
Small utilities useful during training.
"""

from theseus.models import Topology


def find_accumulation_steps(
    batch_size: int, per_device_batch_size: int, topology: Topology
) -> tuple[int, int]:
    """Finds the largest per-device batch size and corresponding number of gradient accumulation steps

    Args:
        batch_size (int): Global batch size
        per_device_batch_size (int): Maximum per-device batch size
        topology (Topology): Topology object containing replica information

    Returns:
        Tuple[int, int]: per-device batch size and number of gradient accumulation steps
    """

    replicas = topology.replicas

    for bs in reversed(range(1, per_device_batch_size + 1)):
        if batch_size % (bs * replicas) == 0:
            return bs, batch_size // (bs * replicas)
    raise ValueError(
        f"No grad_acc found for global_batch_size {batch_size} and max_batch_size {per_device_batch_size} and dp_replicate {replicas}"
    )
