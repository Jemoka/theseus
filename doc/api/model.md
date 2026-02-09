# Model API

## Base Module Abstraction

::: theseus.model.module
    options:
      members:
        - Module
      show_root_heading: false
      show_root_toc_entry: false

## GPT Model

::: theseus.model.models.base
    options:
      members:
        - GPTConfig
        - GPT
      show_root_heading: false
      show_root_toc_entry: false

## Llama Model

::: theseus.model.models.llama
    options:
      members:
        - LlamaConfig
        - Llama
      show_root_heading: false
      show_root_toc_entry: false

## HuggingFace Compatibility Layer

::: theseus.model.huggingface
    options:
      members:
        - HFCompat
      show_root_heading: false
      show_root_toc_entry: false
