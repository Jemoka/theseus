# Jobs and Specs API

## Base Specs

::: theseus.base.job
    options:
      members:
        - JobSpec
        - ExecutionSpec
      show_root_heading: false
      show_root_toc_entry: false

## Job Lifecycle

::: theseus.job
    options:
      members:
        - BasicJob
        - CheckpointedJob
        - RestoreableJob
      show_root_heading: false
      show_root_toc_entry: false
