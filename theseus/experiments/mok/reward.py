import numpy as np
from typing import Any, Dict
from dataclasses import dataclass
from theseus.config import field


@dataclass
class MokConfig:
    weighting: list[float] = field(
        "optimization/mok/weights", default_factory=lambda: [0.5, 0.5]
    )
    eps_min: float = field("optimization/mok/eps_min", default=1e-6)
    eps_max: float = field("optimization/mok/eps_max", default=0.5)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))  # type: ignore


def mok_reward(
    self: Any, evals: Dict[str, np.ndarray], config: MokConfig
) -> np.ndarray:
    """MoK reward placeholder: every rollout gets training-progress in [0, 1].

    Suppose we have a reward vector $r \in \mathbb{R}^n$ and a corresponding weight vector
     $w \in \mathbb{R}^n$. The most common method of reward scalarization for LLM fine-tuning
    is to compute the weighted sum $r_{scalar} = r^{T}w$. 

    We propose an alternative approach to scalarization as follows. First, we scale each component
    of $r$, so that $r \in [0,1]^{n}$. Next, we normalize the weights of $w$, such that $w \in [0,1]^{n}$
    and $\sum_{i=1}^{n}w_i = 1$. Letting $r_w = w \odot r$, we define 
    \\begin{equation}
        \hat{r}_w = \left[
        \\begin{array}{c}
                r_{w1} \\\\
                \\vdots \\\\
                r_{wn} \\\\
                1 - \sum_{i=1}^{n} r_{wi} \\\\
        \end{array}
    \\right]
    \end{equation}
    We can see that $\hat{r}_w$ defines a valid probability distribution, since for all $i$, $0 \leq r_iw_i \leq 1$ and $\sum_{i=1}^nr$. We then define 
    \\begin{equation}
        \hat{w} = 
        \left[
            \\begin{array}{c}
                w_{1} - \\frac{\\varepsilon}{n} \\\\
                \\vdots \\\\
                w_{n} - \\frac{\\varepsilon}{n} \\\\
                \\varepsilon
            \end{array}
        \\right]
    \end{equation}

    For which the same properties hold. 

    To scalarize our reward, we take the KL-divergence between $\hat{r}_w$ and $\hat{w}$: 
    \\begin{equation}
        r_{scalar} = -D_{KL}(\hat{r}_w \| \hat{w})
    \end{equation}

    Both counter and denominator are in micro-batch units (global_step_counter_
    increments by accumulate_steps per optimizer step; total_batches =
    total_steps * accumulate_steps), so micro-vs-batch is consistent.
    """

    assert len(evals) == len(config.weighting), (
        "Number of reward components must match number of weights"
    )

    # mock:
    # evals =  {"chicken": np.array([1.4, 2.5, 3.1]), "egg": np.array([0.2, 0.3, 0.4])}
    # config = MokConfig(weighting=[0.7, 0.3])

    # first normalize
    evals = {key: sigmoid(value) for key, value in evals.items()}  # scale to [0, 1]
    total_weight = sum(config.weighting)
    weights = np.array(config.weighting) / total_weight

    # compute effective epsilon
    progress = min(self.global_step_counter_ / max(self.total_batches, 1), 1.0)
    eps = config.eps_max - (config.eps_max - config.eps_min) * progress

    # compute r_w and w_hat
    r_w = np.array(list(evals.values())) * weights[:, None]
    r_w_hat = np.concatenate([r_w, 1 - r_w.sum(axis=0, keepdims=True)], axis=0)
    w_hat = np.concatenate([weights * (1 - eps), [eps]])

    # compute KL divergence
    kl_div = np.sum(
        r_w_hat * (np.log(r_w_hat + 1e-10) - np.log(w_hat[:, None] + 1e-10)), axis=0
    )
    # reward is negative KL divergence (we want to minimize divergence between r_w_hat and w_hat)
    reward = -kl_div

    return reward  # type: ignore
