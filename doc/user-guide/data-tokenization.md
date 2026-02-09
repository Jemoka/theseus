# Data and Tokenization

Theseus supports two dataset preparation styles and tokenizer-agnostic chat formatting.

## Tokenization Jobs

## `data/tokenize_blockwise_dataset`

Use when your downstream training expects fixed-size, mask-aware samples.

Outputs:

- `<split>.bin` token matrix
- `<split>.bin.mask` boolean mask matrix
- `shape.json`

Padding tokens are masked, and padded labels are set to `-1` downstream.

## `data/tokenize_variable_dataset`

Use for pretraining-style contiguous token streams.

Outputs:

- `train.bin`
- `val.bin`

These are consumed by PMD streaming memmap loader.

## Tokenizer Configuration

Configured under `tokenizer/*`:

- `tokenizer/backend`: `tiktoken` or `huggingface`
- `tokenizer/name`: encoding or tokenizer id
- `tokenizer/huggingface/use_fast`
- `tokenizer/huggingface/use_remote_code`

## Chat Template Behavior

Theseus intentionally enforces deterministic behavior:

- HF backend uses `apply_chat_template`.
- tiktoken backend uses ChatML formatting.

`encode_chat_template(..., tokenize=False)` returns formatted text.

`encode_chat_template(..., tokenize=True)` returns token ids and requires an encoder.

## Example: Blockwise Classification Dataset

```bash
theseus configure data/tokenize_blockwise_dataset mnli_tok.yaml \
  data/dataset=mnli \
  architecture/block_size=512 \
  tokenizer/backend=huggingface \
  tokenizer/name=Qwen/Qwen2.5-0.5B

theseus run tokenize-mnli mnli_tok.yaml /tmp/theseus
```

## Example: Variable Pretraining Dataset

```bash
theseus configure data/tokenize_variable_dataset fineweb_tok.yaml \
  data/dataset=fineweb \
  data/max_samples=1000000 \
  tokenizer/backend=tiktoken \
  tokenizer/name=cl100k_base

theseus run tokenize-fineweb fineweb_tok.yaml /tmp/theseus
```

## Choosing PMD vs Padded at Train Time

In `training/dataset`, each sampling item can set `style`:

- `PMD` for streaming memmap token data
- `PADDED` for fixed matrix + mask data

## Practical Checks

- Verify tokenization outputs before starting large jobs.
- For validation-heavy runs, ensure val split has enough tokens/sequences for configured validation steps.
- Keep tokenizer backend consistent between train/eval where possible.

## Extending Datasets

For a full extension guide (including what to inherit and registry wiring), see:

- `user-guide/writing-datasets-evals.md`
