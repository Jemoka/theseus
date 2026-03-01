"""
mock.py
debugging mocking infrastructure
"""

import functools

import jax
import flax
import jax.numpy as jnp
from flax.linen import Module


def _unwrap_partitioned(v):  # type: ignore[no-untyped-def]
    """Unwrap nn.Partitioned to its inner value if needed."""
    if isinstance(v, flax.linen.Partitioned):
        return v.value
    return v


def _is_static(v) -> bool:  # type: ignore[no-untyped-def]
    """Check if a value is a static (non-traceable) Python value."""
    if isinstance(v, (bool, str, type(None), int, float)):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_static(i) for i in v)
    return False


def _split_kwargs(kwargs):  # type: ignore[no-untyped-def]
    """Split kwargs into static (non-traceable) and dynamic (array) kwargs."""
    static = {k: v for k, v in kwargs.items() if _is_static(v)}
    dynamic = {k: v for k, v in kwargs.items() if k not in static}
    return static, dynamic


def _mock_leaf(leaf, key):  # type: ignore[no-untyped-def]
    """Create a mocked array with the same shape/dtype as `leaf`."""
    # In shape-land, leaves are often ShapeDtypeStruct-like.
    # We assume `.dtype` and `.shape` exist.
    dtype = leaf.dtype
    shape = leaf.shape

    # Floats
    if jnp.issubdtype(dtype, jnp.floating):
        return jax.random.normal(key, shape).astype(dtype)

    # Complex
    if jnp.issubdtype(dtype, jnp.complexfloating):
        k1, k2 = jax.random.split(key)
        real = jax.random.normal(k1, shape)
        imag = jax.random.normal(k2, shape)
        return (real + 1j * imag).astype(dtype)

    # Signed ints
    if jnp.issubdtype(dtype, jnp.signedinteger):
        info = jnp.iinfo(dtype)  # type: ignore[no-untyped-call]
        # Keep range modest to avoid overflow surprises in downstream ops.
        lo = max(info.min, -100)
        hi = min(info.max, 100)
        # randint upper bound is exclusive; ensure hi > lo
        if hi <= lo:
            lo, hi = 0, max(1, int(info.max))
        return jax.random.randint(key, shape, lo, hi, dtype=dtype)

    # Unsigned ints
    if jnp.issubdtype(dtype, jnp.unsignedinteger):
        info = jnp.iinfo(dtype)  # type: ignore[no-untyped-call]
        lo = 0
        hi = min(info.max, 100)
        if hi <= lo:
            hi = max(1, int(info.max))
        return jax.random.randint(key, shape, lo, hi, dtype=dtype)

    # Bool
    if dtype == jnp.bool_ or jnp.issubdtype(dtype, jnp.bool_):
        return jax.random.bernoulli(key, 0.5, shape).astype(jnp.bool_)

    raise ValueError(f"Unsupported dtype {dtype} for mocked value")


def _mock_shape(shape, key=jax.random.PRNGKey(0)):  # type: ignore[no-untyped-def]
    """Mock an output pytree using per-leaf RNG keys."""
    shape = _unwrap_partitioned(shape)  # type: ignore[no-untyped-call]

    leaves, treedef = jax.tree_util.tree_flatten(shape)
    if len(leaves) == 0:
        raise ValueError("init_fn returned an empty pytree")

    # Split RNG per leaf so leaves don't share identical draws.
    keys = jax.random.split(key, len(leaves))

    mocked_leaves = [
        _mock_leaf(leaf, k)  # type: ignore[no-untyped-call]
        for leaf, k in zip(leaves, keys)
    ]
    return jax.tree_util.tree_unflatten(treedef, mocked_leaves)


class InlineMockLinenModule:
    """Helper for inline mocking."""

    def __init__(self, name, obj, key):  # type: ignore[no-untyped-def]
        self.obj = obj
        self.name = name
        self.param_tree = None
        self.key = key

    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        static_kwargs, dynamic_kwargs = _split_kwargs(kwargs)  # type: ignore[no-untyped-call]
        init_fn = functools.partial(self.obj.init, **static_kwargs)
        apply_fn = functools.partial(self.obj.apply, **static_kwargs)

        if self.param_tree is None:
            key, self.key = jax.random.split(self.key)
            self.param_tree = jax.eval_shape(init_fn, key, *args, **dynamic_kwargs)

        output_shape = jax.eval_shape(
            apply_fn, self.param_tree, *args, **dynamic_kwargs
        )

        # IMPORTANT: advance key per call so successive calls differ,
        # while still using per-leaf splits inside _mock_shape.
        out_key, self.key = jax.random.split(self.key)
        return _mock_shape(output_shape, out_key)  # type: ignore[no-untyped-call]

    def __repr__(self):  # type: ignore[no-untyped-def]
        return f"Mocked {self.name}"


class Mocker:
    """Linen-class mocker that enables in-line debugging."""

    def __init__(self) -> None:
        self.key = jax.random.PRNGKey(0)

    def __setattr__(self, name, value):  # type: ignore[no-untyped-def]
        if not isinstance(value, Module):
            super().__setattr__(name, value)
        else:
            key, self.key = jax.random.split(self.key)
            mocked = InlineMockLinenModule(name, value, key)  # type: ignore[no-untyped-call]
            super().__setattr__(name, mocked)

    def param(self, name, init_fn, *init_args, unbox=True, **init_kwargs):  # type: ignore[no-untyped-def]
        # unbox is a dead param
        dummy_rng = jax.random.PRNGKey(0)
        static_kwargs, dynamic_kwargs = _split_kwargs(init_kwargs)  # type: ignore[no-untyped-call]

        def init_wrapper(rng, *dynamic_args):  # type: ignore[no-untyped-def]
            d_iter = iter(dynamic_args)
            full_args = [a if _is_static(a) else next(d_iter) for a in init_args]
            return init_fn(rng, *full_args, **static_kwargs, **dynamic_kwargs)

        dynamic_args = [a for a in init_args if not _is_static(a)]
        init_shape = jax.eval_shape(init_wrapper, dummy_rng, *dynamic_args)

        # Use the Mocker's key so params are reproducible but not constant.
        p_key, self.key = jax.random.split(self.key)
        mocked = _mock_shape(init_shape, p_key)  # type: ignore[no-untyped-call]
        super().__setattr__(name, mocked)
        return mocked
