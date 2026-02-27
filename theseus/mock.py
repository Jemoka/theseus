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


def _mock_leaf(leaf, key=jax.random.PRNGKey(0)):  # type: ignore[no-untyped-def]
    if leaf.dtype == jnp.float32:
        return jax.random.normal(key, leaf.shape).astype(jnp.float32)
    elif leaf.dtype == jnp.bfloat16:
        return jax.random.normal(key, leaf.shape).astype(jnp.bfloat16)
    elif leaf.dtype == jnp.int32:
        return jax.random.randint(key, leaf.shape, 0, 100).astype(jnp.int32)
    elif leaf.dtype == jnp.bool_:
        return jax.random.bernoulli(key, 0.5, leaf.shape).astype(jnp.bool_)
    else:
        raise ValueError(f"Unsupported dtype {leaf.dtype} for mocked value")


def _mock_shape(shape, key=jax.random.PRNGKey(0)):  # type: ignore[no-untyped-def]
    shape = _unwrap_partitioned(shape)  # type: ignore[no-untyped-call]
    leaves = jax.tree.leaves(shape)
    if len(leaves) == 0:
        raise ValueError("init_fn returned an empty pytree")
    elif len(leaves) == 1 and leaves[0] is shape:
        return _mock_leaf(shape, key)  # type: ignore[no-untyped-call]
    return jax.tree.map(functools.partial(_mock_leaf, key=key), shape)


class InlineMockLinenModule:
    """Helper for inline mocking.

    >>> obj = get_an_inline_mock()
    >>> mocked_value = obj(
    ...     mocked,
    ...     inputs,
    ...     here,
    ... )

    You should NEVER, EVER, EVER instantiate
    this object yourself because it will cause subtle
    bugs like "oh no I accidentally left a mocker
    in my code and now it is returning mocked values instead
    of the real ones and I have no idea why"
    """

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
        return _mock_shape(output_shape, self.key)  # type: ignore[no-untyped-call]

    def __repr__(self):  # type: ignore[no-untyped-def]
        return f"Mocked {self.name}"


class Mocker:
    """Linen-class mocker that enables in-line debugging.

    Tip:
    Set this class to `self`, and suddenly you can just
    run parts of this class as if you are using the values
    directly. Useful for inline debugging.
    """

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
            # reconstruct full args in original order, substituting back dynamic ones
            d_iter = iter(dynamic_args)
            full_args = [a if _is_static(a) else next(d_iter) for a in init_args]
            return init_fn(rng, *full_args, **static_kwargs, **dynamic_kwargs)

        dynamic_args = [a for a in init_args if not _is_static(a)]
        init_shape = jax.eval_shape(init_wrapper, dummy_rng, *dynamic_args)
        mocked = _mock_shape(init_shape)  # type: ignore[no-untyped-call]
        super().__setattr__(name, mocked)
        return mocked
