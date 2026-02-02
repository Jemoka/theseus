from typing import Optional
import re
import tiktoken

from theseus.data.datasets import ChatTemplate, ChatTurn


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
    encoder: Optional[tiktoken.Encoding] = None,
    system_prompt: Optional[str] = None,
    prompt: bool = False,
) -> list[int] | str:
    """
    Encode a chat template into tokens using chatml format.

    Format:
    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {message}<|im_end|>
    <|im_start|>assistant
    {message}<|im_end|>

    Args:
        template: List of chat turns
        encoder: Tokenizer (if None, returns the string instead of tokens)
        system_prompt: Optional system prompt to prepend
        prompt: If True, end with <|im_start|>assistant for autoregressive generation
    """
    # Build the full string first
    parts = []

    # Add system prompt if provided
    if system_prompt and system_prompt != "":
        parts.append("<|im_start|>system\n")
        parts.append(system_prompt)
        parts.append("<|im_end|>\n")

    # Add each turn
    for turn in template:
        parts.append(f"<|im_start|>{turn.role}\n")
        parts.append(turn.message)
        parts.append("<|im_end|>\n")

    # Add assistant prompt for autoregression
    if prompt:
        parts.append("<|im_start|>assistant\n")

    # Concatenate
    full_text = "".join(parts)

    # Return string if no encoder, otherwise tokenize
    if encoder is None:
        return full_text

    tokens: list[int] = encoder.encode(full_text, allowed_special="all")
    return tokens


def decode_chat_template(
    tokens: list[int] | str,
    encoder: Optional[tiktoken.Encoding] = None,
) -> ChatTemplate:
    """
    Decode tokens back into a ChatTemplate.

    Parses chatml format:
    <|im_start|>role
    message<|im_end|>

    Args:
        tokens: Token list or string (if encoder is None, treated as string)
        encoder: Tokenizer (if None, tokens is treated as the raw string)
    """
    # Decode tokens to text, or use directly if encoder is None
    if encoder is None:
        text = tokens if isinstance(tokens, str) else "".join(map(str, tokens))
    else:
        text = encoder.decode(tokens)

    # Parse chatml format
    # Pattern: <|im_start|>role\nmessage<|im_end|>
    pattern = r"<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>"
    matches = re.findall(pattern, text, re.DOTALL)

    # Build ChatTemplate, skipping system messages
    template = []
    for role, message in matches:
        role = role.strip()
        message = message.strip()
        template.append(ChatTurn(role=role, message=message))

    return template
