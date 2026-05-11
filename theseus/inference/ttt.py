"""Test-Time-Training inference job.

Extends ``InferenceJob`` to also mutate the ``"fast_weights"`` Flax variable
collection.  ``LaCTBlock._ttt`` declares two variables there — ``W`` (current
fast weights) and ``M`` (optimizer momentum) — and the inner-loop update writes
both back at the end of every forward pass.  By wiring this collection into
``InferenceJob.forward``'s ``mutable`` tuple, the ``_autoregress`` plumbing
inherited from the base threads the mutated state into successive decode steps
for free.

Honesty guarantees (the user's invariant):

- Every ``rollout()`` call starts a fresh prefill with no ``extra_variables``,
  so ``W_var``'s ``init_fn`` runs again against the current slow params
  ``W*_0``.  ⇒ W is reset at the start of every sequence.
- The init_fn closes over the live ``params``, so any change to ``W*_0``
  (gradient step, checkpoint restore) is picked up on the next fresh forward.
  ⇒ W is honestly thrown out whenever outer weights change.

Use this class as the trainer's inference handle for LaCT (and any future
mutable-state architecture that pairs with KV-cache decoding) by overriding the
trainer's ``evaluator()`` factory, or directly when constructing an
``Evaluator``/``InferenceJob`` from a checkpoint.
"""

from typing import Any, Optional, Tuple

import jax

from theseus.inference.base import InferenceJob


class TTTInferenceJob(InferenceJob[Any, Any]):
    """InferenceJob variant that also mutates the ``"fast_weights"`` collection.

    All other behavior — sharding, rollout, autoregressive decoding, padding —
    is inherited unchanged.  The only override is ``forward``, which auto-adds
    ``"fast_weights"`` to ``mutable`` whenever the caller requests any mutation
    (typically the KV ``cache`` during prefill/decode).
    """

    @staticmethod
    def forward(
        state: Any,
        params: Any,
        batch: Tuple[Any, Optional[jax.Array], jax.Array],
        key: Optional[jax.Array] = None,
        deterministic: bool = False,
        mutable: Optional[list[str] | tuple[str, ...]] = None,
        extra_variables: Optional[dict[str, Any]] = None,
        cache_max_len: Optional[int] = None,
    ) -> Any:
        """Forward with auto-paired ``cache`` + ``fast_weights`` mutation.

        We always pair the two collections — there is no LaCT inference path
        that wants KV-cache persistence but not fast-weight persistence (or
        vice versa).  When ``mutable`` is ``None`` (e.g. teacher-forced
        perplexity eval), neither collection is mutated and the model's
        ``_ttt`` branch takes the pure-functional path with ``W = W_0``.
        """
        if mutable is not None:
            seen = tuple(mutable)
            if "fast_weights" not in seen:
                mutable = seen + ("fast_weights",)
            else:
                mutable = seen

        return InferenceJob.forward(
            state,
            params,
            batch,
            key=key,
            deterministic=deterministic,
            mutable=mutable,
            extra_variables=extra_variables,
            cache_max_len=cache_max_len,
        )
