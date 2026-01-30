from theseus.config import field, generate_canonical_config
from dataclasses import dataclass


@dataclass
class Test2:
    boeu: str = field("test")
    bboeu: str = field("oo", default="12")
    boeu: int = field("test/lr", default=1e-3)


@dataclass
class Test:
    moeu: int = field("test/block_size")
    boeu: float = field("test/lr", default=1e-3)


print(generate_canonical_config(Test, Test2))

# fields(Test)[0].type

# for i in fields(Test):
#     print(i.metadata.get("th_config_field"))

# # from theseus.data import PrepareDatasetJob

# # from dataclasses import dataclass, fields


# # @dataclass
# # class Test:
# #     a: str = "field/a"
# #     b: int = "field/b"


# # fields(Test)

# # PrepareDatasetJob.config.model_fields
# # M

# # job = PrepareDatasetJob.local(config, "/Users/houjun/theseus")
# # job
# # # job
# # # # job()
