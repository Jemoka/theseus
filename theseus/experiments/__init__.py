# Jobs are registered via @job decorators in their definition modules.
# See theseus.registry for the authoritative JOBS dict.
from .models.gpt import PretrainGPT  # noqa: F401
from .models.forking import PretrainThoughtbubbles  # noqa: F401
from .models.qwen import PretrainQwen, FinetuneBackboneQwen  # noqa: F401
from .models.llama import PretrainLlama, FinetuneBackboneLlama  # noqa: F401
from .models.gpt_neox import PretrainGPTNeoX, FinetuneBackboneGPTNeoX  # noqa: F401
from .models.sidechannel import (  # noqa: F401
    PretrainSideChannelGPT,
    FinetuneSideChannelGPT,
    FinetuneSideChannelQwen,
)

from .continual import *  # noqa: F401, F403
from .redcodegen import *  # noqa: F401, F403
