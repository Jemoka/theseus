"""
Small utilities useful during training.
"""

from theseus.base import Topology


def estimate_per_device_batch_size(
    chip_memory: int,
    total_params_millions: float,
    shards: int,
    block_size: int,
    vram_calib_factor: float,
) -> int:
    """Estimate max per-device batch size based on VRAM.

    Memory breakdown (per param, bf16 training with AdamW):
    - Params: 2 bytes (bf16)
    - Gradients: 4 bytes (fp32 for accumulation)
    - Master weights: 4 bytes (fp32 copy for optimizer)
    - Optimizer m: 4 bytes (fp32 first moment)
    - Optimizer v: 4 bytes (fp32 second moment)
    Total: ~18 bytes/param (all sharded by tensor parallelism)

    Activations: scales with batch * seq * sqrt(params)

    Args:
        chip_memory: bytes of VRAM per device
        total_params_millions: model parameters in millions
        shards: tensor parallel shards
        block_size: sequence length
        vram_calib_factor: user-tuned calibration for activation memory

    Returns:
        Estimated batch size (at least 1)
    """
    params_per_shard = total_params_millions * 1e6 / shards

    # Fixed memory: ~18 bytes/param (params + grads + optimizer + master weights)
    fixed_memory = params_per_shard * 18
    usable_memory = chip_memory * 0.85 - fixed_memory  # 85% usable

    # Activation memory per sample (empirical scaling)
    bytes_per_sample = vram_calib_factor * block_size * (params_per_shard**0.5)

    estimated = int(usable_memory / bytes_per_sample)
    return max(1, estimated)


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
