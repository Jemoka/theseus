from theseus.config import build, field, hydrate
from typing import List

from dataclasses import dataclass


@dataclass
class Chicken:
    a: int
    b: int


@dataclass
class Test:
    chicken: List[Chicken] = field(
        "rhodesia/chicken", default_factory=lambda: [Chicken(a=12, b=3)]
    )


fg = build(Test)
fg.rhodesia.chicken.append({"a": 4323, "b": 9})

hydrate(Test, fg)

# OmegaConf.to_object(fg)
# OmegaConf.merge(OmegaConf.structured(Test), )
# fg
# hydrate(Test, fg)

# # 1+1


# print(OmegaConf.to_yaml(build(*GPT.dfs())))

# cfg = OmegaConf.create("""
# architecture:
#   n_embd: 128
#   bias: true
#   n_layers: 2
#   vocab_size: 50497
#   block_size: 1024
#   dropout: false
#   n_head: 2
#   rope: true
# """)

# key = jax.random.PRNGKey(7)
# key, key_init = jax.random.split(key)

# inputs = jnp.ones((32, 1024)).astype(int)

# with configuration(cfg):
#     model = configure(GPT)
#     vars = model.init(key_init, inputs)


# # from typing import List, Tuple
# # from dataclasses import dataclass


# # @dataclass
# # class Chicken:
# #     name: str = field("data/dataset")
# #     chob: str = field("chombai")


# # @dataclass
# # class TokenizeDatasetConfigBase:
# #     """Base config for dataset tokenization"""

# #     name: str = field("data/dataset")
# #     suffix: str = field("data/suffix", default="")
# #     val_pct: float = field("data/val_pct", default=0.05)
# #     seed: int = field("data/seed", default=2357)
# #     dataset: List[Tuple[str, float]] = field(
# #         "data/thing", default_factory=lambda: [("fineweb", 1.0)]
# #     )


# # res = TokenizeDatasetConfigBase("te")
# # cfg = build(res)
# # cfg.data.dataset = "12"


# # def make_thing():
# #     return configure(TokenizeDatasetConfigBase)


# # with configuration(cfg):
# #     thing = make_thing()

# # thing
# # cfg
