from typing import Optional
import tiktoken

from theseus.data.datasets import ChatTemplate


def get_chatml_encoder() -> tiktoken.Encoding:
    """Create tiktoken encoder with chatml special tokens"""
    cl100k_base = tiktoken.get_encoding("cl100k_base")

    enc = tiktoken.Encoding(
        name="cl100k_im",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            "<|im_start|>": 100264,
            "<|im_end|>": 100265,
        },
    )
    return enc


def encode_chat_template(
    template: ChatTemplate,
    encoder: tiktoken.Encoding,
    system_prompt: Optional[str] = None,
) -> list[int]:
    """
    Encode a chat template into tokens using chatml format.

    Format:
    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {message}<|im_end|>
    <|im_start|>assistant
    {message}<|im_end|>
    """
    tokens = []

    # Add system prompt if provided
    if system_prompt:
        tokens.extend(encoder.encode("<|im_start|>system\n"))
        tokens.extend(encoder.encode(system_prompt))
        tokens.extend(encoder.encode("<|im_end|>\n"))

    # Add each turn
    for turn in template:
        tokens.extend(encoder.encode(f"<|im_start|>{turn.role}\n"))
        tokens.extend(encoder.encode(turn.message))
        tokens.extend(encoder.encode("<|im_end|>\n"))

    return tokens
