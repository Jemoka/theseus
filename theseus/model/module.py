"""
A theseus nn module, which is nn module, execpt it asks you to
dump all the types of constitutent parts so we can configure it.
"""

from abc import abstractproperty

from typing import Type, List, Any, Tuple, Optional
from flax.linen import Module as nn


class Module(nn):
    @abstractproperty
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        """Return the sharding configuration for this module.

        Returns:
            Tuple[Tuple[str, Optional[Axis]]]: A dictionary mapping Axes to sharding dimensions.
        """

        ...

    @classmethod
    def components(cls) -> List[Type[Any]]:
        """Return the types of constituent parts of this module.

        Returns:
            Type: A type or tuple of types representing the constituent parts.
        """

        raise NotImplementedError(
            "To configure a module, you must implement components()"
        )

    @classmethod
    def gather(cls) -> List[Type[Any]]:
        """Depth-first search of all constituent parts of this module.

        Returns:
            List[Type]: A list of all constituent part types.
        """

        parts: List[Type[Any]] = [cls]
        for component in cls.components():
            if issubclass(component, Module):
                parts.extend(component.gather())
            else:
                parts.append(component)
        return list(set(parts))
