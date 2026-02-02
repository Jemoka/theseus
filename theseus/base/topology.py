"""
Hardware information and topology representation for distributed JAX setups.
Defines a Topology class that encapsulates device and process information,
as well as JAX Mesh configuration.
"""

import jax
import numpy as np
from jax.sharding import Mesh

from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict

from theseus.base.axis import Axis
from theseus.base.chip import Chip


class Topology(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chip: Annotated[Chip, Field(description="what device is used")]
    device_count: Annotated[int, Field(description="number of devices across cluster")]
    local_device_count: Annotated[int, Field(description="number of devices locally")]
    process_count: Annotated[
        int, Field(description="number of processes running across cluster")
    ]
    is_main: Annotated[
        bool, Field(description="whether this process is the main process")
    ]

    mesh: Annotated[
        Mesh,
        Field(description="JAX Mesh representing the device topology", exclude=True),
    ]

    replicas: Annotated[
        int, Field(description="number of SPMD replicas across cluster")
    ]
    local_replicas: Annotated[
        int, Field(description="number of SPMD replicas per host")
    ]

    @classmethod
    def new(cls, chip: Chip, shard_into: int | None = None) -> "Topology":
        """Create a Topology instance based on the current JAX device configuration.

        Args:
            chip: The chip type being used.
            shard_into: Number of shards to divide the devices into for tensor parallelism.
                        If None, defaults to local device count (shard evenly within each host).
                        The SPMD/data parallel axis is determined automatically.

        """
        devs = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
        local = jax.local_device_count()

        # Default to sharding by local device count if not specified
        if shard_into is None:
            shard_into = local

        devices = np.array(devs).reshape(-1, local)
        devices = devices.reshape(-1, shard_into)

        mesh = Mesh(devices, (Axis.BATCH, Axis.SHARD))

        replicas = jax.device_count() // shard_into
        local_replicas = local // shard_into

        assert replicas == mesh.shape[Axis.BATCH]
        assert local_replicas * jax.process_count() == replicas

        return cls(
            chip=chip,
            device_count=mesh.size,
            local_device_count=local,
            process_count=jax.process_count(),
            is_main=jax.process_index() == 0,
            mesh=mesh,
            replicas=replicas,
            local_replicas=local_replicas,
        )
