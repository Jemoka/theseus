from dataclasses import dataclass
from typing import Any, List

import jax
from loguru import logger

from theseus.config import field
from theseus.data.tokenizer import get_tokenizer
from theseus.inference.base import InferenceJob
from theseus.model.models.base import GPT
from theseus.registry import job


@dataclass
class ChatConfig:
    block_size: int = field("architecture/block_size", default=512)
    per_device_batch_size: int = field("training/per_device_batch_size", default=-1)
    max_new_tokens: int = field("inference/max_new_tokens", default=256)
    temperature: float = field("inference/temperature", default=0.8)
    top_p: float = field("inference/top_p", default=0.9)


@job("gpt/debug/chat/continuation")
class Chat(InferenceJob[ChatConfig, GPT]):
    MODEL = GPT

    @classmethod
    def config(cls) -> List[Any]:
        return [ChatConfig]

    def run(self) -> None:
        tok = get_tokenizer()
        key = jax.random.PRNGKey(42)

        max_new = self.args.max_new_tokens
        temp = self.args.temperature
        top_p = self.args.top_p

        logger.info(
            "CHAT | ready  max_new_tokens={}  temperature={}  top_p={}",
            max_new,
            temp,
            top_p,
        )
        print("\n--- scratchbubbles chat (ctrl-c to quit) ---\n")

        while True:
            try:
                prompt = input("> ")
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                break

            if not prompt.strip():
                continue

            ids = tok.encode(prompt)
            xs, masks = self.pad([ids])
            prompt_len = xs.shape[-1]
            total_tokens = min(prompt_len + max_new, self.block_size)

            key, subkey = jax.random.split(key)
            result = self._autoregress(
                self.state,
                subkey,
                xs,
                masks,
                total_tokens,
                temp,
                top_p,
            )

            gen_ids = result[0, prompt_len:].tolist()
            # strip trailing eot
            eot = tok.eot_token
            if eot in gen_ids:
                gen_ids = gen_ids[: gen_ids.index(eot)]

            print(tok.decode(gen_ids))
            print()
