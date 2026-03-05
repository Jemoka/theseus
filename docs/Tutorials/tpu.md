# TPU Dispatch Guide

Theseus can dispatch jobs to Google Cloud TPU VMs. It manages the full
lifecycle: creating the TPU VM if it doesn't exist, shipping code to all
workers, and launching the job across the pod.

## Prerequisites

Before using the TPU backend you need:

1. **The `gcloud` CLI installed and authenticated.**
   ```bash
   gcloud auth login
   gcloud config set project my-project
   ```

2. **TPU quota** in your GCP project for the accelerator type you want
   (e.g. `v4-32`, `v5e-16`). Check your quotas in the
   [GCP Console](https://console.cloud.google.com/iam-admin/quotas).

3. **A GCP zone** with TPU availability. Common zones:
   - `us-central2-b` (v4)
   - `us-east1-d` (v5e)
   - `us-east5-b` (v5p)

   Check what's available:
   ```bash
   gcloud compute tpus accelerator-types list --zone=us-central2-b
   ```

## Dispatch Config

Add a TPU host to your `~/.theseus.yaml`. Here is a walkthrough of every
field:

```yaml
clusters:
  # Paths on the TPU VM filesystem.
  # These are local to the VM, not a shared filesystem.
  gcp:
    root: /home/user/theseus-data     # data, checkpoints
    work: /home/user/theseus-work     # scratch/working directory
    log: /home/user/theseus-logs      # log directory

hosts:
  # The host name is used as the TPU VM name in gcloud commands.
  # Pick something descriptive — this is what shows up in `gcloud compute tpus list`.
  my-tpu-v4:
    type: tpu

    # Must match a cluster entry above.
    cluster: gcp

    # GCP zone where the TPU will be created.
    zone: us-central2-b

    # GCP project. If omitted, uses your gcloud default project.
    project: my-gcp-project

    # TPU accelerator type. Format: "v{version}-{chips}".
    # The number is the total chip count across the pod.
    # Examples: "v4-8" (single host), "v4-32" (4 hosts x 8 chips),
    #           "v5e-16", "v5p-128"
    accelerator_type: v4-32

    # TPU software/runtime version.
    # List available versions:
    #   gcloud compute tpus versions list --zone=us-central2-b
    version: tpu-ubuntu2204-base

    # --- Pricing options (pick one or neither) ---

    # Spot VMs: cheaper, but GCP can preempt at any time.
    spot: true

    # Preemptible: cheaper, 24h time limit, can be preempted.
    # preemptible: false

    # --- Optional fields ---

    # VPC network and subnetwork (if your project uses custom networking).
    # network: my-vpc
    # subnetwork: my-subnet

    # GCP service account for the TPU VM.
    # service_account: sa@proj.iam.gserviceaccount.com

    # Use internal IP for SSH/SCP (required in some VPC setups).
    # internal_ip: false

    # Instance metadata key-value pairs.
    # metadata:
    #   startup-script: "echo hello"

    # uv dependency groups to sync in the bootstrap script.
    uv_groups: [tpu]

priority:
  - my-tpu-v4
```

## Submitting a Job

```bash
# Basic submit — solver matches your chip request against the TPU host:
theseus submit my-run experiment.yaml --chip tpu-v4 -n 32

# If you only have one TPU host, chip/n flags are optional:
theseus submit my-run experiment.yaml

# Override TPU software version:
theseus submit my-run experiment.yaml --tpu-version tpu-vm-v4-base

# Use spot pricing (override config):
theseus submit my-run experiment.yaml --tpu-spot

# Use on-demand pricing (override config):
theseus submit my-run experiment.yaml --tpu-on-demand

# Include uncommitted changes:
theseus submit my-run experiment.yaml --dirty
```

## How It Works Under the Hood

When you run `theseus submit` targeting a TPU host:

1. **Solver** matches your hardware request against the TPU host's
   `accelerator_type`. The chip name is parsed from the accelerator type
   (e.g. `v4-32` → chip `tpu-v4`, 32 chips).

2. **TPU VM lifecycle**: If the TPU VM doesn't exist, theseus prompts you
   for confirmation (creating a TPU incurs GCP costs), then creates it via
   `gcloud compute tpus tpu-vm create` and waits for it to reach `READY`
   state. If it already exists and is `READY`, this step is skipped.

3. **Code shipping**: Your repo is `git archive`'d into a tarball, SCP'd
   to **all workers** in the TPU pod via `gcloud compute tpus tpu-vm scp
   --worker=all`, then extracted in-place. This ensures identical code on
   every host.

4. **Bootstrap scripts**: The bootstrap shell script and Python dispatch
   script(s) are SCP'd to all workers the same way.

5. **Launch**: The bootstrap script is executed on **all workers**
   simultaneously via `gcloud compute tpus tpu-vm ssh --worker=all`. Each
   worker runs `uv sync --group tpu`, then executes the dispatch Python
   script. JAX's `jax.distributed.initialize()` coordinates the workers
   into a single pod.

6. **Done**: The job runs in the background via `nohup`. Logs are written
   to the `log` directory on the TPU VM.

## Monitoring Jobs

TPU jobs run as background processes on the VM. To check on them:

```bash
# SSH into worker 0:
gcloud compute tpus tpu-vm ssh my-tpu-v4 --zone=us-central2-b --worker=0

# Tail the log file:
tail -f /home/user/theseus-logs/<project>_<group>_<name>_<timestamp>.log

# Check if the process is still running:
gcloud compute tpus tpu-vm ssh my-tpu-v4 --zone=us-central2-b --worker=0 \
  --command="ps aux | grep python"
```

## Deleting TPU VMs

TPU VMs incur costs while they exist, even when idle. Delete them when
you're done:

```bash
gcloud compute tpus tpu-vm delete my-tpu-v4 --zone=us-central2-b --quiet
```

List all your TPU VMs:

```bash
gcloud compute tpus tpu-vm list --zone=us-central2-b
```

## Troubleshooting

### TPU VM creation fails with quota error

You've hit your TPU quota. Check and request increases in the
[GCP Console](https://console.cloud.google.com/iam-admin/quotas).
Filter by `TPU` to find the relevant quota for your accelerator type.

### "PREEMPTED" or "TERMINATED" state

Spot and preemptible TPUs can be reclaimed by GCP at any time.
Re-submit the job — theseus will recreate the TPU VM automatically.

### SSH connection fails

- Check that `gcloud auth login` is current.
- If using `internal_ip: true`, make sure you're on the same VPC or
  have a VPN/IAP tunnel configured.
- Verify the TPU VM is in `READY` state:
  ```bash
  gcloud compute tpus tpu-vm describe my-tpu-v4 --zone=us-central2-b
  ```

### JAX doesn't see all chips

- Verify code was shipped to **all** workers (check that the work
  directory exists on each worker).
- Check that `jax.distributed.initialize()` is being called. The
  bootstrap sets `THESEUS_TPU_MODE=1` so the training code knows
  to initialize distributed.

### Wrong TPU software version

List available versions and pick one that matches your JAX version:
```bash
gcloud compute tpus versions list --zone=us-central2-b
```
