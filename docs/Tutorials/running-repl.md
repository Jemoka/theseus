# Interactive REPL

Launch a Jupyter notebook session on remote hardware instead of a batch job. You will need a [dispatch config](running-remote.md#prerequisites-theseusyaml) with a host that has `uv_groups` configured for Jupyter support.

## Launch a session

```bash
theseus repl --chip h100 -n 1
```

Theseus allocates a node, starts Jupyter, and SSH-forwards the port back to `localhost:8888`. Open that URL in your browser and you're in.

## Live code sync

If your dispatch config includes a `mount` or `proxy` key, you can push local edits to a running REPL session without restarting it.

Launch with sync enabled:

```bash
theseus repl --chip h100 -n 1 --sync
```

Then, after editing files locally, push the changes:

```bash
theseus repl --update
```

Only files tracked by git are synced. Uncommitted edits are included.
