"""
Quick smoke test for the RedCodeGen hardening contrastive dataset.

Requires the example JSONL referenced in the dataset file:
    /Users/houjun/Downloads/qwen-contrastive-2pairs.jsonl
"""

from pathlib import Path

from theseus.data.datasets.redcodegen.hardening import RCGHardeningDataset
from theseus.data.tokenizer import TokenizerConfig, get_tokenizer
from theseus.data.tokenize import (
    TokenizeContrastiveDatasetConfig,
    _build_padded_arrays,
    _encode_dataset_item,
)


def main() -> None:
    data_path = Path("/Users/houjun/Downloads/qwen-contrastive-2pairs.jsonl")
    assert data_path.exists(), f"Missing sample file at {data_path}"

    dataset = RCGHardeningDataset(config=str(data_path))
    assert len(dataset) > 0, "Dataset loaded but is empty"

    tokenizer = get_tokenizer(TokenizerConfig(backend="tiktoken", name="cl100k_base"))
    # Name must exist in registry; we only need the other fields for helpers.
    args = TokenizeContrastiveDatasetConfig(
        name="fever",
        split="noop",
        block_size=64,
        pad_token=0,
        assistant_only=True,
        system_prompt="",
    )

    pos, neg = dataset[0]

    pos_tokens, pos_mask, pos_len, _, _ = _build_padded_arrays(
        *_encode_dataset_item(pos, True, tokenizer, args),
        args.block_size,
        args.pad_token,
    )
    neg_tokens, neg_mask, neg_len, _, _ = _build_padded_arrays(
        *_encode_dataset_item(neg, True, tokenizer, args),
        args.block_size,
        args.pad_token,
    )

    assert pos_tokens.shape == (args.block_size,)
    assert neg_tokens.shape == (args.block_size,)
    assert pos_mask.shape == (args.block_size,)
    assert neg_mask.shape == (args.block_size,)
    assert pos_len > 0 and neg_len > 0
    assert pos_tokens.dtype == neg_tokens.dtype

    print("Smoke test passed:")
    print(f"  block_size: {args.block_size}")
    print(f"  pos_len(raw): {pos_len}, neg_len(raw): {neg_len}")
    print(
        f"  pos_mask true count: {pos_mask.sum()}, neg_mask true count: {neg_mask.sum()}"
    )
    print("  tokenizer backend: tiktoken chatml")

    # Quick round-trip decoding to ensure tokens are valid under tokenizer
    pos_decoded = tokenizer.decode(pos_tokens.tolist())
    neg_decoded = tokenizer.decode(neg_tokens.tolist())
    sample_text = "hello world, contrastive tokenization!"
    sample_tokens = tokenizer.encode(sample_text)
    sample_decoded = tokenizer.decode(sample_tokens)

    print("\nRound-trip checks (truncated to 160 chars):")
    pos_preview = pos_decoded[:160].replace("\n", " ")
    neg_preview = neg_decoded[:160].replace("\n", " ")
    print(f"  pos decoded: {pos_preview}")
    print(f"  neg decoded: {neg_preview}")
    print(f"  sample text -> tokens -> decode: '{sample_decoded}'")


if __name__ == "__main__":
    main()
